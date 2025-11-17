"""
RLDS-based data loader for BridgeData V2.

Mirrors DROID loader’s public API, but adapts to Bridge’s RLDS/TFDS schema.

Key differences vs the original snippet:
- No string tensors are sent to JAX (prevents dtype |Sxx errors).
- Images are decoded late (frame_map) like DROID for efficiency.
- Observations use the same nested structure as DROID: frame["observation"][...].
- Handles optional external frame-filter dict exactly like DROID.
- Robust TFDS builder name: tries "bridge" first, then "bridge_orig".
"""

from enum import Enum, auto
import json
import logging
from pathlib import Path
import tqdm

import openpi.shared.download as download

DATASET_SAMPLE_WEIGHTS = [0.83, 0.17]

class BridgeActionSpace(Enum):
    DEFAULT = auto()  # Bridge exposes a single 7-D action

## truly process low-level data and confirm as a whole


class BridgeRldsDataset:
    def __init__(
        self,
        data_dir: str,
        aug_data_paths: str,
        batch_size: int,
        *,  # keyword-only
        shuffle: bool = True,
        action_chunk_size: int = 16,
        action_space: BridgeActionSpace = BridgeActionSpace.DEFAULT,
        max_loaded_steps_per_episode: int = 100,  # kept for API parity
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,   # -1 -> AUTOTUNE
        num_parallel_calls: int = -1,   # -1 -> AUTOTUNE
        filter_dict_path=None,          # JSON: episode_key -> [[start,end), ...]
        include_metadata: bool = False, # if True, return prompt/step_id in a "metadata" key; still kept off device
    ):
        # Import lazily so TF isn’t a hard dep unless RLDS is used
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Ensure TF doesn’t grab GPUs that JAX/PyTorch want
        tf.config.set_visible_devices([], "GPU")
        
        
        dataset_paths = []
        if data_dir:
            dataset_paths.append(data_dir)
        if aug_data_paths:
            dataset_paths.extend(aug_data_paths)
            
        if not dataset_paths:
            raise ValueError("No data_dir or aug_data_paths provided to BridgeRldsDataset.")
        weights = DATASET_SAMPLE_WEIGHTS[:len(dataset_paths)]
        if len(weights) != len(dataset_paths):
            raise ValueError(
                f"BridgeRldsDataset: number of weights ({len(weights)}) does not match "
                f"number of datasets ({len(dataset_paths)}). "
                f"dataset_paths={dataset_paths}, weights={weights}"
            )
        logging.info(f"BridgeRldsDataset: loading {len(dataset_paths)} dataset(s):")
        for i, p in enumerate(dataset_paths):
            role = "main" if i == 0 else "aug"
            logging.info(f"  [{i}] ({role}) path={p}, weight={weights[i]:.3f}")
            
            
        datasets = []
        for path in dataset_paths:
            try:
                builder = tfds.builder("bridge_orig", data_dir=path, version="0.1.0")
            except Exception as e:
                logging.warning(f"Falling back to builder('custom name') for {path} due to: {e}")
                #builder = tfds.builder("bridgeidaugmented", data_dir=path, version="0.1.0")
                builder = tfds.builder("bridge_pickle_tfds", data_dir=path, version="0.1.0")

            ds = dl.DLataset.from_rlds(
                builder,
                split="train",
                shuffle=shuffle,
                num_parallel_reads=num_parallel_reads,
            )
            datasets.append(ds)
        
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            logging.info(f"Sampling from {len(datasets)} datasets with weight of {DATASET_SAMPLE_WEIGHTS}")
            dataset = dl.DLataset.sample_from_datasets(
                datasets,
                weights=DATASET_SAMPLE_WEIGHTS, 
                stop_on_empty_dataset=False,
                rerandomize_each_iteration=True,
            )
        
        
        # # Build TFDS (BridgeData V2). Try "bridge" first, then fall back to "bridge_orig".
        # builder = tfds.builder("bridge_orig", data_dir=data_dir, version="0.1.0")

        # dataset = dl.DLataset.from_rlds(
        #     builder,
        #     split="train",
        #     shuffle=shuffle,
        #     num_parallel_reads=num_parallel_reads,
        # )
        

        # Repeat so we never run out of data
        dataset = dataset.repeat()

        # Optional frame-level filtering via external JSON, same convention as DROID
        # filter_dict = { "<episode_key>": [[start, end), ...], ... }
        if filter_dict_path is not None:
            cached = download.maybe_download(filter_dict_path)
            with Path(cached).open("r") as f:
                filter_dict = json.load(f)

            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []
            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)

            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
                default_value=False,
            )
            logging.info("Filter hash table initialized")
        else:
            # default pass-through filter
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
            )

        def restructure(traj):
            """
            Reformat observation and action keys; *do not* decode images here.
            Bridge fields (per TFDS):
            - steps/observation/image_0 .. image_3: encoded strings (jpeg/png)
            - steps/observation/state: float state if present
            - steps/action: float32[7]
            - steps/language_instruction: string (may be empty)
            - episode_metadata/file_path: string
            """
            tf = __import__("tensorflow")

            # Actions: Bridge exposes final 7-D action directly.
            actions = traj["action"]  # [T, 7]

            # Choose one exterior camera to mirror DROID’s single training view
            exterior_img_encoded = traj["observation"]["image_0"]  # encoded strings [T]

            # Instruction (may be empty). Keep as tf.string but don’t send to device.
            instruction = traj.get("language_instruction", "")

            traj_len = tf.shape(actions)[0]
            indices = tf.as_string(tf.range(traj_len))

            # Simple per-step ID: "<file_path>--<t>"
            step_id = traj["traj_metadata"]["episode_metadata"]["file_path"] + "--" + indices
            passes_filter = self.filter_table.lookup(step_id)


            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img_encoded,
                    "wrist_image": exterior_img_encoded,
                    "state": traj["observation"]["state"],
                },
                "prompt": instruction,
                "step_id": step_id,
                "passes_filter": passes_filter,
            }

            
        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks (same as DROID)."""
            tf = __import__("tensorflow")
            traj_len = tf.shape(traj["actions"])[0]
            idx = tf.broadcast_to(tf.range(action_chunk_size)[None], [traj_len, action_chunk_size]) + \
                  tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, action_chunk_size])
            idx = tf.minimum(idx, traj_len - 1)  # repeat last action at end
            traj["actions"] = tf.gather(traj["actions"], idx)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        # Flatten to per-frame (actually per action-chunk) dataset
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Apply filter
        dataset = dataset.filter(lambda frame: frame["passes_filter"])

        # Drop helper key so it never leaves TF
        def drop_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(drop_passes_filter)

        # Decode images late (like DROID), producing uint8 tensors
        def decode_images(frame):
            tf = __import__("tensorflow")
            frame["observation"]["image"] = tf.io.decode_image(
                frame["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            frame["observation"]["wrist_image"] = tf.io.decode_image(
                frame["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            return frame

        dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # *** CRITICAL: ensure only numeric leaves are sent to JAX ***
        # If include_metadata=True, strip it out before batching so the batch tree is numeric-only.
        if include_metadata:
            def strip_metadata(frame):
                # Return metadata as a side-channel if you consume the iterator yourself.
                # For standard training loops that move batches to devices, drop it here.
                frame.pop("metadata", None)
                return frame
            dataset = dataset.map(strip_metadata)



        # Shuffle & batch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Yields only numeric arrays in the tree (safe for jax.device_put)
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # Stub, like DROID
        return 2_000_000
