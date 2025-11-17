# from https://github.com/HaomingSong/openpi/blob/main/src/openpi/policies/bridge_pad_policy.py
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_bridge_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # "observation/left_yellow_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # "observation/right_blue_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # "observation/wirst_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # breakpoint()
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BridgePadInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0  # We don't mask for pi0-FAST.

        # NOTE: for bridge dataset at IPEC-COMMUNITY/bridge_orig_lerobot, the state is 8-dim.
        # Get the state. We are padding from 8 to the model action dim.
        state = data["observation/state"][:8]
        # state = torch.zeros(data["observation/state"].shape)

        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # breakpoint()
        # breakpoint()

        if "observation/image" in data:
            primary_image = _parse_image(data["observation/image"])
        elif "observation/primary_image" in data:
            primary_image = _parse_image(data["observation/primary_image"])
        elif "observation/image_0" in data:
            primary_image = _parse_image(data["observation/image_0"])
        else:
            raise KeyError(
                "No valid image key found. Expected one of: "
                "'observation/image', 'observation/primary_image', or 'observation/image_0'."
            )
        # left_yellow_image = _parse_image(data["observation/left_yellow_image"])
        # right_blue_image = _parse_image(data["observation/right_blue_image"])
        # wrist_image = _parse_image(data["observation/wrist_image"])
        # breakpoint()
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": primary_image,
                # "left_yellow_image": left_yellow_image,
                "left_wrist_0_rgb": np.zeros_like(primary_image),
                # "right_blue_image": right_blue_image,
                "right_wrist_0_rgb": np.zeros_like(primary_image),
                # "wrist_image": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BridgePadOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}
