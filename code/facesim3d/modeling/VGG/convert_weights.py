"""
Convert `VGG-Face` weights from `LuaTorch` to `PyTorch`.

!!! note
    This is an adaptation of: https://github.com/chi0tzp/PyVGGFace/blob/master/convert_weights.py.

Author: Simon M. Hofmann
Years: 2023
"""

# %% Import
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import torch
import torchfile
from ut.ils import cprint

from facesim3d.configs import paths
from facesim3d.modeling.VGG.models import VGGFace

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
WEIGHT_FILE: str = "VGG_FACE.t7"


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def download_torch_weights() -> Path:
    """Download `torch` weights for `VGG-Face` if necessary."""
    # Create output directory (if necessary)
    p2_model = Path(paths.data.models.vggface).parent
    p2_model.mkdir(parents=True, exist_ok=True)

    p2_torch_weights = Path(paths.data.models.vggface, WEIGHT_FILE)

    if not p2_torch_weights.is_file():
        torch_tar_file = Path(paths.data.models.vggface).parent / "vgg_face_torch.tar.gz"
        # Download tar.gz file
        cprint(string="Downloading VGG-Face torch weights 'vgg_face_torch.tar.gz'...", col="b")
        urllib.request.urlretrieve(
            url="http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz", filename=torch_tar_file
        )

        # Extract weight file from tar.gz
        if not p2_torch_weights.is_file():
            cprint(string="Extracting weights from 'vgg_face_torch.tar.gz'...", col="b")
            with tarfile.open(torch_tar_file) as tar_file:
                tar_file.extract(member=f"vgg_face_torch/{WEIGHT_FILE}", path=p2_model)
            torch_tar_file.unlink()

    return p2_torch_weights


def convert_weights(path_to_weights: str | Path, model: VGGFace) -> VGGFace:
    """
    Convert `LuaTorch` weights and load them into a `PyTorch` model.

    :param path_to_weights: filename of the pre-trained `LuaTorch` weights file [str]
    :param model: `VGGFace` model
    :return: `VGGFace` model with loaded weights
    """
    torch_model = torchfile.load(path_to_weights)
    counter = 1
    block = 1
    block_size = [2, 2, 3, 3, 3]
    block_cut: int = 5
    for _i, layer in enumerate(torch_model.modules):
        if layer.weight is not None:
            if block <= block_cut:
                self_layer = model.features[f"conv_{block}_{counter}"]
                counter += 1
                if counter > block_size[block - 1]:
                    counter = 1
                    block += 1
                self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
            else:
                self_layer = model.fc[f"fc{block}"]
                block += 1
                self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
    return model


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Download weights if necessary
    p2_torch_weights_file = download_torch_weights()

    # Define VGGFace instance
    print("Convert original pre-trained LuaTorch and load them to VGGFace model...")
    vggface_model = VGGFace()
    vggface_model = convert_weights(path_to_weights=p2_torch_weights_file, model=vggface_model)

    # Save output model
    p2_vggface_model = Path(paths.data.models.vggface, "vggface.pth")
    cprint(string=f"Save VGGFace weights at {p2_vggface_model} ... ", col="b")
    torch.save(vggface_model.state_dict(), p2_vggface_model)

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
