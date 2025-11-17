"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        def is_str_like(v):
            a = np.asarray(v)
            return np.issubdtype(a.dtype, np.str_) or np.issubdtype(a.dtype, np.bytes_)
        def strip(tree):
            if isinstance(tree, dict):
                out = {}
                for k, v in tree.items():
                    if isinstance(v, (dict, list, tuple)):
                        sv = strip(v)
                        # keep containers even if empty; dataloader can handle it
                        out[k] = sv
                    else:
                        if not is_str_like(v):
                            out[k] = v
                return out
            elif isinstance(tree, (list, tuple)):
                return type(tree)(strip(v) for v in tree)
            else:
                return tree if not is_str_like(tree) else None  # will get dropped by parent
        # Drop any None values that may appear from lists/tuples
        cleaned = strip(x)
        if isinstance(cleaned, dict):
            cleaned = {k: v for k, v in cleaned.items() if v is not None}
        return cleaned


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    print('no bug before creating dataloader')
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    print("[DEBUG] built data_loader; num_batches =", num_batches, flush=True)
    it = iter(data_loader)
    print("[DEBUG] pulling first batch...", flush=True)
    first = next(it)   # <- if it hangs here, it’s the dataset/transform/IO
    print("[DEBUG] got first batch keys:", list(first.keys()), flush=True)

    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    print('finded the config')
    data_config = config.data.create(config.assets_dirs, config.model)
    print('data_config loading successfully')
    if data_config.rlds_data_dir is not None:
        print('creat rlds data loaders')
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        print('create torch data loader')
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )
    print('nothing wrong in creating dataloader')
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    print('no error with main')
    tyro.cli(main)
