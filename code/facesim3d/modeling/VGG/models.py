"""
Collection comprising adaptations of the `VGG-Face` model.

!!! info "Model sources"
    ▸ https://www.robots.ox.ac.uk/~vgg/software/vgg_face/ (original model weights but in LuaTorch)

    ▸ https://github.com/chi0tzp/PyVGGFace (most is adopted from here)

    ▸ https://modelzoo.co/model/facenet-pytorch

    ▸ https://www.kaggle.com/code/shubhendumishra/recognizing-faces-in-the-wild-vggface-pytorch

    ▸ Note there is also a second version of VGGFace https://github.com/ox-vgg/vgg_face2

"""

# %% Import
from __future__ import annotations

from abc import ABC, abstractmethod
from ast import literal_eval
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchsummary import summary
from torchviz import make_dot
from ut.ils import browse_files, cinput, cprint, deprecated

from facesim3d.configs import params, paths

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
VALID_DECISION_BLOCK_MODES = [
    "fc",  # fully connected layers: a number of tests showed that this does not converge (2023-11-07)
    "conv",  # decision block based on convolutional layers (this works, 2023-11-07)
]

# %% Functions & classes < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def h_out(h_in, k, s, p, d=1):
    """
    Calculate the output height of a convolutional layer.

    :param h_in: input height
    :param k: kernel size (height)
    :param s: stride
    :param p: padding
    :param d: dilation
    :return: output height.
    """
    return int((h_in + 2 * p - d * (k - 1) - 1) / s + 1)


def w_out(w_in, k, s, p, d=1):
    """
    Calculate the output width of a convolutional layer.

    :param w_in: input width
    :param k: kernel size (width)
    :param s: stride
    :param p: padding
    :param d: dilation
    :return: output width.
    """
    return int((w_in + 2 * p - d * (k - 1) - 1) / s + 1)


def check_exclusive_gender_trials(exclusive_gender_trials: str | None) -> str | None:
    """Check the variable `exclusive_gender_trials`, which is used in different functions."""
    if exclusive_gender_trials is not None:
        msg = f"exclusive_gender_trials must be in {params.GENDERS} OR None."
        if isinstance(exclusive_gender_trials, str):
            exclusive_gender_trials = exclusive_gender_trials.lower()
            if exclusive_gender_trials not in params.GENDERS:
                raise ValueError(msg)
        else:
            raise TypeError(msg)
    return exclusive_gender_trials


@lru_cache(maxsize=1)
def get_vgg_layer_names() -> list[str]:
    """Return a list of layer names constituting the `VGGFace` model."""
    return VGGFace(save_layer_output=False).layer_names


@lru_cache(maxsize=1)
def read_vgg_layer_table() -> pd.DataFrame:
    """Read the table with `VGG` layer names and corresponding output shapes, and number of parameters."""
    return pd.read_csv(
        filepath_or_buffer=paths.data.models.vgg.output_shapes,
        sep="\t",
        header=0,
        converters={"output_shape": literal_eval},
    )


def get_vgg_layer_feature(layer_name: str, feature: str = "output_shape") -> list[..., int] | int:
    """
    Get the output shape of a given `VGG` layer.

    :param layer_name: Name of the layer.
    :param feature: Feature to return, either 'output_shape' or 'n_params', or so (see table columns)
    :return: layer feature
    """
    layer_name = layer_name.lower()
    layer_tab = read_vgg_layer_table()
    if layer_name not in layer_tab.layer_names.to_numpy():
        msg = f"Layer '{layer_name}' not found in {paths.data.models.vgg.output_shapes}."
        raise ValueError(msg)
    feature = feature.lower()
    if feature not in layer_tab.columns:
        msg = f"Feature '{feature}' not found in {layer_tab.columns.to_list()}."
        raise ValueError(msg)
    return layer_tab.loc[layer_tab.layer_names == layer_name][feature].to_list()[0]


class VGGFace(nn.Module):
    """
    VGGFace class.

    This is an reimplementation of the original `VGG-Face` model in `PyTorch`.

    *Source: https://github.com/chi0tzp/PyVGGFace/blob/master/lib/vggface.py.*
    """

    def __init__(self, save_layer_output: bool = False) -> None:
        """
        Initialize VGGFace model.

        :param save_layer_output: If True, save the output of each layer in a list.
        :return: None
        """
        super().__init__()

        self.save_layer_output = save_layer_output
        self.layer_output = []
        self._layer_names = []

        self.features = nn.ModuleDict(
            OrderedDict(
                {
                    # === Block 1 ===
                    "conv_1_1": nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                    "relu_1_1": nn.ReLU(inplace=True),
                    "conv_1_2": nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    "relu_1_2": nn.ReLU(inplace=True),
                    "maxp_1_2": nn.MaxPool2d(kernel_size=2, stride=2),
                    # === Block 2 ===
                    "conv_2_1": nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                    "relu_2_1": nn.ReLU(inplace=True),
                    "conv_2_2": nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                    "relu_2_2": nn.ReLU(inplace=True),
                    "maxp_2_2": nn.MaxPool2d(kernel_size=2, stride=2),
                    # === Block 3 ===
                    "conv_3_1": nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                    "relu_3_1": nn.ReLU(inplace=True),
                    "conv_3_2": nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    "relu_3_2": nn.ReLU(inplace=True),
                    "conv_3_3": nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    "relu_3_3": nn.ReLU(inplace=True),
                    "maxp_3_3": nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                    # === Block 4 ===
                    "conv_4_1": nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                    "relu_4_1": nn.ReLU(inplace=True),
                    "conv_4_2": nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    "relu_4_2": nn.ReLU(inplace=True),
                    "conv_4_3": nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    "relu_4_3": nn.ReLU(inplace=True),
                    "maxp_4_3": nn.MaxPool2d(kernel_size=2, stride=2),
                    # === Block 5 ===
                    "conv_5_1": nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    "relu_5_1": nn.ReLU(inplace=True),
                    "conv_5_2": nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    "relu_5_2": nn.ReLU(inplace=True),
                    "conv_5_3": nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    "relu_5_3": nn.ReLU(inplace=True),
                    "maxp_5_3": nn.MaxPool2d(kernel_size=2, stride=2),
                }
            )
        )

        self.fc = nn.ModuleDict(
            OrderedDict(
                {
                    "fc6": nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                    "fc6-relu": nn.ReLU(inplace=True),
                    "fc6-dropout": nn.Dropout(p=0.5),
                    "fc7": nn.Linear(in_features=4096, out_features=4096),
                    "fc7-relu": nn.ReLU(inplace=True),
                    "fc7-dropout": nn.Dropout(p=0.5),
                    "fc8": nn.Linear(in_features=4096, out_features=2622),
                }
            )
        )

    def reset_layer_output(self):
        """Reset the layer output list (i.e., set it to an empty list)."""
        self.layer_output = []

    @property
    def layer_names(self):
        """Return list of layer names in `VGGFace`."""
        if self._layer_names:
            return self._layer_names

        for child in self.children():
            for layer in child:
                self._layer_names.append(str(layer))
        return self._layer_names

    def forward(self, x):
        """Run forward pass through the model `VGGFace`."""
        if self.save_layer_output:
            self.reset_layer_output()

        # Forward through feature layers
        for layer in self.features.values():
            x = layer(x)
            # Append layer output to list
            if self.save_layer_output:
                self.layer_output.append(x)

        # Flatten convolution outputs
        x = x.view(x.size(0), -1)

        # Forward through FC layers
        if hasattr(self, "fc"):  # check for later cut-off models
            for layer in self.fc.values():
                x = layer(x)
                if self.save_layer_output:
                    self.layer_output.append(x)

        return x


def load_trained_vgg_weights_into_model(model: VGGFace) -> VGGFace:
    """Load trained weights into the original `VGG-Face` model."""
    model_dict = torch.load(
        Path(paths.data.models.vggface, "vggface.pth"),
        map_location=lambda storage, loc: storage,  # noqa: ARG005
    )
    model.load_state_dict(model_dict)
    return model


def get_vgg_face_model(save_layer_output: bool) -> VGGFace:
    """Get the originally trained `VGGFace` model."""
    vgg_model = VGGFace(save_layer_output=save_layer_output).double()
    return load_trained_vgg_weights_into_model(vgg_model)


def create_fc_bridge(last_core_layer: str) -> nn.ModuleDict | None:
    """Build a bridge between the `VGG core` and the decision block with fully connected layers."""
    # Get layer parameters
    in_features = np.prod(get_vgg_layer_feature(last_core_layer))  # note, this raises error if layer not found
    # in_features: for conv layer this is, e.g., (bs, 512, 7, 7) -> 25088 (here: "maxp_5_3"); for fc this is -> 4096
    last_block_nr = next(int(c) for c in last_core_layer if c.isdigit())  # e.g., "maxp_5_3" -> 5; "fc6" -> 6

    # Create bridge
    if last_block_nr in range(1, 5 + 1):  # in one of the conv blocks, e.g., "maxp_5_3"
        # For conv blocks
        return nn.ModuleDict(
            OrderedDict(
                {
                    f"fc{last_block_nr + 1}": nn.Linear(in_features=in_features, out_features=4096),
                    f"fc{last_block_nr + 1}-relu": nn.ReLU(inplace=True),
                    f"fc{last_block_nr + 1}-dropout": nn.Dropout(p=0.5, inplace=False),
                }
            )
        )

    # i.e., if "fc" in last_core_layer: # in {"fc6-relu", "fc6-dropout", "fc7-relu", "fc7-dropout"}
    # For fc layers
    return nn.ModuleDict(
        OrderedDict(
            {
                f"fc{last_block_nr + 1}": nn.Linear(in_features=in_features, out_features=300),
                f"fc{last_block_nr + 1}-relu": nn.ReLU(inplace=True),
                f"fc{last_block_nr + 1}-dropout": nn.Dropout(p=0.5),
            }
        )
    )


@deprecated(message="Deprecated: FC decision block does not converge. Use create_conv_decision_block() instead.")
def create_fc_decision_block(last_core_layer: str) -> nn.ModuleDict:
    """Build a decision block with fully connected (fc) layers."""
    # Get layer parameters
    last_block_nr = next(int(c) for c in last_core_layer if c.isdigit())  # e.g., "maxp_5_3" -> 5; fc6 -> 6

    if last_block_nr in range(1, 5 + 1):  # in one of the conv blocks, e.g., "maxp_5_3"
        # For conv blocks
        return nn.ModuleDict(
            OrderedDict(
                {
                    f"fc_d_{last_block_nr + 2}": nn.Linear(in_features=4096 * 3, out_features=1024),
                    f"fc_d_{last_block_nr + 2}-relu": nn.ReLU(inplace=True),
                    f"fc_d_{last_block_nr + 2}-dropout": nn.Dropout(p=0.5, inplace=False),
                    f"fc_d_{last_block_nr + 3}": nn.Linear(in_features=1024, out_features=3),  # final layer
                }
            )
        )
    if "fc" in last_core_layer:  # in {"fc6-relu", "fc6-dropout", "fc7-relu", "fc7-dropout"}
        # For fc layers
        return nn.ModuleDict(
            OrderedDict({f"fc_d_{last_block_nr + 2}": nn.Linear(in_features=300 * 3, out_features=3)})
        )

    # Error if layer not implemented
    msg = f"Cut layer '{last_core_layer}' not implemented yet for create_fc_decision_block()."
    raise NotImplementedError(msg)


def create_conv_decision_block(last_core_layer: str) -> nn.ModuleDict:
    """Build a decision block with convolutional layers only."""
    last_core_layer = last_core_layer.lower()
    # Get layer parameters
    last_block_nr = next(int(c) for c in last_core_layer if c.isdigit())  # e.g., "maxp_5_3" -> 5; "fc6" -> 6

    if last_block_nr in range(1, 5 + 1):
        # the core model was cut off at its conv block
        return nn.ModuleDict(
            OrderedDict(
                {  # in (bs, 1, 6, 4096)
                    f"conv_d_{last_block_nr + 2}_1": nn.Conv2d(
                        in_channels=1, out_channels=2, kernel_size=(2, 50), padding=0, stride=(2, 1)
                    ),  # -> (bs, 2, 3, 4047)
                    f"relu_d_{last_block_nr + 2}_1": nn.ReLU(inplace=True),
                    f"conv_d_{last_block_nr + 2}_2": nn.Conv2d(
                        in_channels=2, out_channels=2, kernel_size=(3, 100), padding=0, stride=(1, 2)
                    ),  # -> (bs, 2, 1, 1974)
                    f"relu_d_{last_block_nr + 2}_2": nn.ReLU(inplace=True),
                    f"conv_d_{last_block_nr + 2}_3": nn.Conv2d(
                        in_channels=2, out_channels=1, kernel_size=(1, 100), padding=0, stride=2
                    ),  # -> (bs, 1, 1, 938)
                    f"relu_d_{last_block_nr + 2}_3": nn.ReLU(inplace=True),
                    f"conv_d_{last_block_nr + 2}_4": nn.Conv2d(
                        in_channels=1, out_channels=1, kernel_size=(1, 100), padding=0, stride=3
                    ),  # -> (bs, 1, 1, 280)
                    f"relu_d_{last_block_nr + 2}_4": nn.ReLU(inplace=True),
                    f"conv_d_{last_block_nr + 2}_5": nn.Conv2d(
                        in_channels=1, out_channels=1, kernel_size=(1, 80), padding=0, stride=3
                    ),  # -> (bs, 1, 1, 67)
                    f"relu_d_{last_block_nr + 2}_5": nn.ReLU(inplace=True),
                    f"conv_d_{last_block_nr + 3}_1": nn.Conv2d(
                        in_channels=1, out_channels=1, kernel_size=(1, 65), padding=0, stride=1
                    ),  # -> (bs, 1, 1, 3); final layer
                }
            )
        )
    if "fc" in last_core_layer:  # in {"fc6-relu", "fc6-dropout", "fc7-relu", "fc7-dropout"}
        return nn.ModuleDict(
            OrderedDict(
                {  # in (bs, 1, 6, 300)
                    f"conv_d_{last_block_nr + 2}_1": nn.Conv2d(
                        in_channels=1, out_channels=2, kernel_size=(2, 50), padding=0, stride=(2, 1)
                    ),  # -> (bs, 2, 3, 251)
                    f"relu_d_{last_block_nr + 2}_1": nn.ReLU(inplace=True),
                    f"conv_d_{last_block_nr + 2}_2": nn.Conv2d(
                        in_channels=2, out_channels=2, kernel_size=(3, 100), padding=0
                    ),  # -> (bs, 2, 1, 152)
                    f"relu_d_{last_block_nr + 2}_2": nn.ReLU(inplace=True),
                    f"conv_d_{last_block_nr + 2}_3": nn.Conv2d(
                        in_channels=2, out_channels=1, kernel_size=(1, 100), padding=0
                    ),  # -> (bs, 1, 1, 53)
                    f"relu_d_{last_block_nr + 2}_3": nn.ReLU(inplace=True),
                    f"conv_d_{last_block_nr + 3}_1": nn.Conv2d(
                        in_channels=1, out_channels=1, kernel_size=(1, 51), padding=0
                    ),  # -> (bs, 1, 1, 3)
                }
            )
        )

    # Error if layer not implemented
    msg = f"Cut layer '{last_core_layer}' not implemented yet for create_conv_decision_block()."
    raise NotImplementedError(msg)


class VGGcore(nn.Module):
    """
    The `VGGcore` class is used to extract a core part of the `VGGFace` model.

    For this, the original `VGGFace` is cut off at a given layer.
    """

    def __init__(self, freeze_vgg_core: bool, last_core_layer: str, verbose: bool = False) -> None:
        """Initialize the `VGGcore` model."""
        super().__init__()
        self.vgg_core = get_vgg_face_model(save_layer_output=False)
        self._layer_names = []
        # Keep only layers until maxp_5_3
        self.last_core_layer = last_core_layer
        self._cut_off(last_core_layer=last_core_layer, verbose=verbose)
        self.freeze_vgg_core = freeze_vgg_core
        if self.freeze_vgg_core:
            self._freeze_vgg_core()

    def _cut_off(self, last_core_layer: str, verbose: bool = True):
        """Cut off layers until last_core_layer."""
        if last_core_layer not in self.layer_names:
            msg = f"Layer '{last_core_layer}' not found in {self.__class__.__name__}.layer_names."
            raise ValueError(msg)

        if "-dropout" in last_core_layer:
            # Replace dropout layer with the previous relu layer
            self.last_core_layer = last_core_layer = last_core_layer.replace("-dropout", "-relu")
            cprint(
                string="Cutting at the dropout layer is not possible!\n"
                f"Instead, the model will be cut at the previous layer: '{last_core_layer}'",
                col="y",
            )

        # Iterate backwards through network and delete layers until last_core_layer
        self._layer_names = []  # reset layer names
        cut_done = False
        for name, child in reversed(list(self.vgg_core.named_children())):
            if not isinstance(child, torch.nn.modules.container.ModuleDict):
                msg = f"Child '{name}' is not a ModuleDict, but {type(child)}."
                raise TypeError(msg)
            for layer_name, _ in reversed(list(child.named_children())):
                if layer_name == last_core_layer:
                    cut_done = True
                    break
                if verbose:
                    cprint(string=f"Deleting layer '{layer_name}' from {name}-module.", col="y")
                delattr(child, layer_name)
            if cut_done:
                break

        # Remove empty modules dict
        for module_name, child in reversed(list(self.vgg_core.named_children())):
            if isinstance(child, torch.nn.modules.container.ModuleDict) and len(child) == 0:
                if verbose:
                    cprint(string=f"Deleting empty {module_name}-module.", col="y")
                delattr(self.vgg_core, module_name)

        if verbose:
            cprint(string="\nRemaining layers:", col="b", fm="ul")
            print("", *self.layer_names, sep="\n\t")

    def _freeze_vgg_core(self):
        for param in self.vgg_core.parameters():
            param.requires_grad = False

    @property
    def layer_names(self):
        """Return a list of layer names in the model."""
        if self._layer_names:
            return self._layer_names

        for child in self.children():
            if isinstance(child, VGGFace):
                for grandchild in child.children():
                    for layer_name in grandchild:
                        self._layer_names.append(str(layer_name))
            else:
                for layer_name in child:
                    self._layer_names.append(str(layer_name))

        return self._layer_names

    def forward(self, x):
        """Run the forward pass through the `VGGcore` model."""
        return self.vgg_core(x)


class VGGMultiView(VGGcore):
    """Original `VGG-Face` model retrained to predict face IDs from multiple views."""

    def __init__(
        self,
        freeze_vgg_core: bool,
        last_core_layer: str = "fc7-relu",
        n_face_ids: int = params.main.n_faces,  # 100 in the main study
        verbose: bool = False,
    ) -> None:
        """Initialize the `VGGMultiView` model."""
        if "fc" not in last_core_layer:
            msg = f"Last core layer must be one of the FC layers, not '{last_core_layer}'."
            # Could cut at an earlier layer, too, but this would need to be implemented
            raise ValueError(msg)
        super().__init__(freeze_vgg_core=freeze_vgg_core, last_core_layer=last_core_layer, verbose=verbose)
        self.verbose = verbose

        # Create decision layer above the last core layer of VGGFace
        self.decision_layer = nn.ModuleDict(
            OrderedDict(
                {
                    "fc_d": nn.Linear(
                        in_features=self.find_output_dims_of_last_core_layer(),  # get out dims of the last core layer
                        out_features=n_face_ids,
                    )
                }
            )
        )
        self.decision_layer.apply(self.init_weights)  # might not be necessary

    def find_output_dims_of_last_core_layer(self):
        """Find the output dimensions of the last core layer."""
        n_features = None
        for name, child in reversed(list(self.vgg_core.named_children())):
            if not isinstance(child, torch.nn.modules.container.ModuleDict):
                msg = f"Child '{name}' is not a ModuleDict, but {type(child)}."
                raise TypeError(msg)
            for layer_name, layer in reversed(list(child.named_children())):
                if layer_name == self.last_core_layer.split("-")[0]:  # ignore dropout & relu layers
                    n_features = layer.out_features
                    if self.verbose:
                        cprint(
                            string=f"Found output dims ({n_features}, 1) of layer '{self.last_core_layer}' "
                            f"from the {name}-module.",
                            col="y",
                        )
                    break
            if n_features is not None:
                break
        if n_features is None:
            msg = f"Could not find output dimensions of layer '{self.last_core_layer}'."
            raise ValueError(msg)
        return n_features

    @staticmethod
    def init_weights(m):
        """Initialize the model weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.01)  # torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """Run the forward pass through the model."""
        x = self.vgg_core(x)
        for layer in self.decision_layer.values():
            # keep iteration for future extensions
            x = layer(x)
        return x


class VGGFaceHumanjudgmentBase(nn.Module, ABC):
    """
    Base class for the `VGG-Face` model for human similarity judgments.

    * the architecture consists of three parallel `VGG-Face` models.
    * for each trial, each VGG submodel gets one of the faces from the triplet, respectively.
    * weights are shared between the three models
    * we combine the outputs of the three models with linear layer(s) to predict the human choice in the trial

    This is similar to: https://github.com/pytorch/examples/blob/main/siamese_network/main.py

    This base class builds the body for two different variants of the model for human similarity judgments:
    * VGGFaceHumanjudgment: trained on face images directly. It directs data from VGGcore -> VGGFaceHumanjudgmentBase
    * VGGFaceHumanjudgmentFrozenCore: trained on activation maps of VGGFace in layer 'maxp_5_3'
    """

    def __init__(
        self,
        decision_block: str,
        freeze_vgg_core: bool,
        last_core_layer: str,
        parallel_bridge: bool,
        session: str | None,
    ) -> None:
        """Initialize VGGFaceHumanjudgmentBase."""
        super().__init__()

        self._layer_names = []  # initialize
        self.freeze_vgg_core: bool = freeze_vgg_core
        self.last_core_layer = last_core_layer
        self.parallel_bridge = parallel_bridge
        self.session: str | None = session

        # Replace the last core layer of VGGFace
        if self.parallel_bridge:
            self.vgg_core_bridge = nn.ModuleDict(
                OrderedDict(
                    {
                        "bridge1": create_fc_bridge(last_core_layer=self.last_core_layer),
                        "bridge2": create_fc_bridge(last_core_layer=self.last_core_layer),
                        "bridge3": create_fc_bridge(last_core_layer=self.last_core_layer),
                    }
                )
            )

        else:
            self.vgg_core_bridge = create_fc_bridge(last_core_layer=self.last_core_layer)

        # Create decision block
        self.decision_block_mode = decision_block

        # Set decision block, too
        if self.decision_block_mode == "fc":
            self.decision_block = create_fc_decision_block(last_core_layer=self.last_core_layer)
            self.decision_block.apply(self.init_weights)  # might not be necessary
        elif self.decision_block_mode == "conv":
            self.decision_block = create_conv_decision_block(last_core_layer=self.last_core_layer)

    @staticmethod
    def init_weights(m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.01)  # torch.nn.init.zeros_(m.bias)

    @property
    def last_core_layer(self) -> str:
        """Return the cut layer of the `VGGcore`, i.e., the last layer before the `bridge` is attached."""
        return self._last_core_layer

    @last_core_layer.setter
    def last_core_layer(self, layer_name: str):
        """Set core cut layer."""
        # Check type
        if not isinstance(layer_name, str):
            msg = f"Core cut layer must be str, not {type(layer_name)}"
            raise TypeError(msg)

        # Prepare name
        layer_name = layer_name.lower()

        # Check for dropout layer
        if "-dropout" in layer_name:
            # Replace dropout layer with the previous relu layer
            layer_name = layer_name.replace("-dropout", "-relu")
            cprint(
                string="Cutting at the dropout layer is not possible!\n"
                f"Instead, the model will be cut at the previous layer: '{layer_name}'",
                col="y",
            )

        # Check if layer exists
        if layer_name not in get_vgg_layer_names():
            msg = f"Layer '{layer_name}' not found in VGGFace model."
            raise ValueError(msg)

        self._last_core_layer = layer_name

    @property
    def vgg_core_bridge(self):
        """Return `VGG core bridge`."""
        return self._vgg_core_bridge

    @vgg_core_bridge.setter
    def vgg_core_bridge(self, module: nn.ModuleDict) -> None:
        """Set VGG core bridge."""
        if not isinstance(module, nn.ModuleDict):
            msg = f"VGG core bridge must be nn.ModuleDict, not {type(module)}"
            raise TypeError(msg)

        self._vgg_core_bridge = module
        self._vgg_core_bridge.float()  # this layer is not frozen, even if freeze_vgg_core is True

    @property
    def decision_block_mode(self) -> str:
        """Return the decision block mode."""
        return self._decision_block_mode

    @decision_block_mode.setter
    def decision_block_mode(self, mode: str) -> None:
        """Set decision block mode."""
        if not isinstance(mode, str):
            msg = f"Decision block mode must be str, not {type(mode)}"
            raise TypeError(msg)
        mode = mode.lower()
        if mode not in VALID_DECISION_BLOCK_MODES:
            msg = f"Decision block mode must be one of {VALID_DECISION_BLOCK_MODES}, not '{mode}'!"
            raise ValueError(msg)

        self._decision_block_mode = mode

    @property
    def decision_block(self) -> nn.ModuleDict:
        """Return the decision block of the model."""
        return self._decision_block

    @decision_block.setter
    def decision_block(self, module: nn.ModuleDict) -> None:
        """Set decision block."""
        if not isinstance(module, nn.ModuleDict):
            msg = f"Decision block must be nn.ModuleDict, not {type(module)}"
            raise TypeError(msg)
        self._decision_block = module

    @property
    def layer_names(self) -> list[str]:
        """Return a list of layer names in the model."""
        if self._layer_names:
            return self._layer_names

        for child in self.children():
            if isinstance(child, (VGGFace, VGGcore)):  # noqa: UP038
                self._layer_names.extend(child.layer_names)
            else:
                for layer in child:
                    self._layer_names.append(str(layer))
        return self._layer_names

    def requires_grad(self, layer_name: str | None = None) -> None:
        """Return whether a layer requires a gradient flow."""
        found = False  # initialize
        for param_name, param in self.named_parameters():
            if layer_name is None or layer_name in param_name:
                print(f"'{param_name}' requires grad: {param.requires_grad}")
                found = True
        if not found:
            cprint(string=f"There is no layer containing parameters with the name '{layer_name}'.", col="r")

    @abstractmethod
    def forward_vgg(self, x: torch.Tensor, bridge_idx: int | None) -> torch.Tensor:
        """Run the forward pass through `VGG core` part of the model."""

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        """Run the forward pass through the whole model."""
        x1 = self.forward_vgg(x1, bridge_idx=1 if self.parallel_bridge else None)
        x2 = self.forward_vgg(x2, bridge_idx=2 if self.parallel_bridge else None)
        x3 = self.forward_vgg(x3, bridge_idx=3 if self.parallel_bridge else None)

        # Reshape for decision block
        if self.decision_block_mode == "fc":
            x = torch.stack((x1, x2, x3), dim=1)
            x = x.view(x.size(0), -1)  # (batch_size, 4096 * 3)
        elif self.decision_block_mode == "conv":
            x = torch.stack((x1, x2, x1, x3, x2, x3), dim=1)  # all 6 combinations
            x = x.unsqueeze(1)  # add channel dimensions (batch_size, channel=1, combs=6, x[1|2|3].shape=300|4096)
        else:
            msg = f"Decision block mode '{self.decision_block_mode}' not implemented (yet)."
            raise NotImplementedError(msg)

        # Forward through decision layers
        for layer in self.decision_block.values():
            x = layer(x)

        # Reshape before release
        if self.decision_block_mode == "conv":
            x = x.view(x.size(0), -1)

        return x


class VGGFaceHumanjudgment(VGGFaceHumanjudgmentBase):
    """
    An adaptation of the `VGG-Face` model for human similarity judgments.

    The `VGGFaceHumanjudgment` model consists of three parallel face models (based on `VGGcore`):

    * For each trial, each submodel gets one of the face images which are part of the corresponding triplet.
    * The outputs of the three models are combined with linear layer(s) (`FC bridge`).
    * Weights are shared between the three submodels at the bottom (`VGGcore` + `bridge`).
    * Then the concatenated feature maps are pushed through a `decision block` to predict human choices in a trial.

    *Compare to: https://github.com/pytorch/examples/blob/main/siamese_network/main.py*
    """

    def __init__(
        self,
        decision_block: str,
        freeze_vgg_core: bool,
        last_core_layer: str,
        parallel_bridge: bool = False,
        session: str | None = None,
    ) -> None:
        """Initialize the `VGGFaceHumanjudgment` model."""
        super().__init__(
            decision_block=decision_block,
            freeze_vgg_core=freeze_vgg_core,
            last_core_layer=last_core_layer,
            parallel_bridge=parallel_bridge,
            session=session,
        )
        self.vgg_core = VGGcore(freeze_vgg_core=freeze_vgg_core, last_core_layer=last_core_layer, verbose=False)
        self.vgg_core.float()  # float32 for GPU

    def forward_vgg(self, x: torch.Tensor, bridge_idx: int | None) -> torch.Tensor:
        """Run the forward pass through the `VGGcore` and then through layers of the `VGGFaceHumanjudgmentBase`."""
        x = self.vgg_core(x)
        x = x.view(x.size(0), -1)  # or ...size()[0], -1), this flattens the tensor
        # The bridge(s) come(s) from VGGFaceHumanjudgmentBase
        if self.parallel_bridge:
            for layer in self.vgg_core_bridge[f"bridge{bridge_idx}"].values():
                x = layer(x)
        else:
            for layer in self.vgg_core_bridge.values():
                x = layer(x)
        return x


class VGGFaceHumanjudgmentFrozenCore(VGGFaceHumanjudgmentBase):
    """
    An adaptation of the `VGG-Face` model for human similarity judgments, where the `VGG core` is frozen.

    This model is similar to the `VGGFaceHumanjudgment` variant, however,
    the model gets pre-computed activation maps of a given layer (`last_core_layer`) of `VGG-Face` as input.

    These activation maps, from three faces in a trial, are concatenated and pushed through a `decision block`.
    """

    def __init__(
        self, decision_block: str, last_core_layer: str, parallel_bridge: bool = False, session: str | None = None
    ) -> None:
        """Initialize model."""
        super().__init__(
            decision_block=decision_block,
            freeze_vgg_core=True,
            last_core_layer=last_core_layer,
            parallel_bridge=parallel_bridge,
            session=session,
        )

    def forward_vgg(self, x: torch.Tensor, bridge_idx: int | None) -> torch.Tensor:
        """Run the forward pass through `VGG bridge(s)`."""
        # The bridge(s) come(s) from VGGFaceHumanjudgmentBase
        if self.parallel_bridge:
            for layer in self.vgg_core_bridge[f"bridge{bridge_idx}"].values():
                x = layer(x)
        else:
            for layer in self.vgg_core_bridge.values():
                x = layer(x)
        return x


@deprecated(message="Use VGGFaceHumanjudgmentFrozenCore instead.")
class VGGFaceHumanjudgmentFrozenCoreOld(nn.Module):
    """
    Old, that is, deprecated frozen-core `VGG-Face` model for human similarity judgments.

    The model comprises:

    * The model takes the activation maps of the `VGG-Face` in layer `"fc7-relu"`
    * It gets three activation maps representing three faces, and it feets them to the same `"fc8"` layer
    * Then the output is passed through a decision block.
    """

    def __init__(self, decision_block: str) -> None:
        """Initialize the `VGGFaceHumanjudgmentFrozenCoreOld` model."""
        super().__init__()

        self._layer_names = []
        self.freeze_vgg_core = True
        self.vgg_core_fc8 = nn.ModuleDict(
            OrderedDict(
                {
                    "fc8": nn.Linear(in_features=4096, out_features=300),  # replace last layer
                    "fc8-relu": nn.ReLU(inplace=True),
                    "fc8-dropout": nn.Dropout(p=0.5),
                }
            )
        )
        self.vgg_core_fc8.float()  # float32 for GPU, since the input is smaller now, we could use float64
        self.decision_block_mode = decision_block.lower()
        if self.decision_block_mode not in {"fc", "conv"}:
            msg = "Decision block mode must be 'fc' or 'conv'."
            raise ValueError(msg)
        if self.decision_block_mode == "fc":
            self.decision_block = nn.ModuleDict(
                OrderedDict(
                    {
                        "fc_d_9": nn.Linear(in_features=300 * 3, out_features=300),
                        "fc_d_9-relu": nn.ReLU(inplace=True),
                        "fc_d_9-dropout": nn.Dropout(p=0.5),
                        "fc_d_10": nn.Linear(in_features=300, out_features=3),
                    }
                )
            )
            # Initialize weights
            self.decision_block.apply(self.init_weights)  # might not be necessary

        elif self.decision_block_mode == "conv":
            self.decision_block = nn.ModuleDict(
                OrderedDict(
                    {  # in (bs, 1, 6, 300)
                        "conv_d_9_1": nn.Conv2d(
                            in_channels=1, out_channels=2, kernel_size=(2, 50), padding=0, stride=(2, 1)
                        ),  # -> (bs, 2, 3, 251)
                        "relu_d_9_1": nn.ReLU(inplace=True),
                        "conv_d_9_2": nn.Conv2d(
                            in_channels=2, out_channels=2, kernel_size=(3, 100), padding=0
                        ),  # -> (bs, 2, 1, 152)
                        "relu_d_9_2": nn.ReLU(inplace=True),
                        "conv_d_9_3": nn.Conv2d(
                            in_channels=2, out_channels=1, kernel_size=(1, 100), padding=0
                        ),  # -> (bs, 1, 1, 53)
                        "relu_d_9_3": nn.ReLU(inplace=True),
                        "conv_d_10_1": nn.Conv2d(
                            in_channels=1, out_channels=1, kernel_size=(1, 51), padding=0
                        ),  # -> (bs, 1, 1, 3)
                    }
                )
            )

    @staticmethod
    def init_weights(m):
        """Initialize the weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.01)  # torch.nn.init.zeros_(m.bias)

    @property
    def layer_names(self):
        """Return a list of layer names in the model."""
        if self._layer_names:
            return self._layer_names
        for child in self.children():
            for layer in child:
                self._layer_names.append(str(layer))
        return self._layer_names

    def forward_vgg(self, x):
        """Run the forward pass through the `VGG core`."""
        # Forward through FC layers
        for layer in self.vgg_core_fc8.values():
            x = layer(x)
        return x

    def forward(self, x1, x2, x3):
        """Run the forward pass."""
        x1 = self.forward_vgg(x1)
        x2 = self.forward_vgg(x2)
        x3 = self.forward_vgg(x3)

        # Reshape for decision block
        if self.decision_block_mode == "fc":
            x = torch.stack((x1, x2, x3), dim=1)
            x = x.view(x.size(0), -1)  # (batch_size, 300 * 3)
        elif self.decision_block_mode == "conv":
            x = torch.stack((x1, x2, x1, x3, x2, x3), dim=1)  # all 6 combinations
            x = x.unsqueeze(1)  # add channel dimension,  (batch_size, channel=1, combs=6, 300)
        else:
            msg = "Decision block mode must be 'fc' or 'conv'."
            raise ValueError(msg)

        # Forward through decision layers
        for layer in self.decision_block.values():
            x = layer(x)

        if self.decision_block_mode == "conv":
            # Reshape before release
            x = x.view(x.size(0), -1)

        return x


class VGGFaceHumanjudgmentFrozenCoreWithLegs(VGGFaceHumanjudgment):
    """
    A model extension to feed face images to the `VGGFaceHumanjudgmentFrozenCore`.

    That is, the model is fed with whole images,
    instead of activation maps from the last cut layer of the `VGGFace core`.

    This is used to apply XAI methods upon `VGGFaceHumanjudgmentFrozenCore` to find relevant areas in the input images
    that drive the decision of the model (see `facesim3d.modeling.VGG.explain.py`).
    """

    def __init__(self, frozen_top_model: VGGFaceHumanjudgmentFrozenCore) -> None:
        """Initialize the `VGGFaceHumanjudgmentFrozenCoreWithLegs` model."""
        super().__init__(
            decision_block=frozen_top_model.decision_block_mode,
            freeze_vgg_core=frozen_top_model.freeze_vgg_core,
            last_core_layer=frozen_top_model.last_core_layer,
            session=frozen_top_model.session,
        )
        self._frozen_top_model = frozen_top_model
        self.frozen_top_model_name = frozen_top_model.name
        self.vgg_core_bridge = frozen_top_model.vgg_core_bridge  # overwrite vgg_core_bridge
        self.decision_block = frozen_top_model.decision_block  # overwrite decision_block

    @property
    def frozen_top_model(self):
        """Return the frozen top-model."""
        if self._frozen_top_model is None:
            print(f"Loading frozen top model '{self.frozen_top_model_name}' from memory ...")

            self._frozen_top_model = load_trained_vgg_face_human_judgment_model(
                session=self.session, model_name=self.frozen_top_model_name, device=None
            )
        return self._frozen_top_model

    @frozen_top_model.setter
    def frozen_top_model(self, value: VGGFaceHumanjudgmentFrozenCore):
        """Set the frozen top-model."""
        if not isinstance(value, VGGFaceHumanjudgmentFrozenCore):
            msg = (
                f"VGGFaceHumanjudgmentFrozenCoreWithLegs.frozen_top_model must be VGGFaceHumanjudgmentFrozenCore, "
                f"not {type(value)}"
            )
            raise TypeError(msg)
        self._frozen_top_model = value

    @property
    def layer_names(self):
        """Return the layer names of the model."""
        if self._layer_names:
            return self._layer_names
        for child in self.children():
            if isinstance(child, VGGcore):
                self._layer_names.extend(child.layer_names)
            else:
                for layer in child:
                    self._layer_names.append(str(layer))
        # TODO: could be sorted  # noqa: FIX002
        return self._layer_names


def load_trained_vgg_face_human_judgment_model(
    session: str,
    model_name: str | None = None,
    exclusive_gender_trials: str | None = None,
    device: str | None = None,
) -> VGGFaceHumanjudgment | VGGFaceHumanjudgmentFrozenCore:
    """Load a trained `VGGFaceHumanjudgment` model from a file."""
    exclusive_gender_trials = check_exclusive_gender_trials(exclusive_gender_trials=exclusive_gender_trials)
    g_sfx = "" if exclusive_gender_trials is None else f"{exclusive_gender_trials}_only_trials"
    p2_model_root = Path(paths.data.models.vggbehave, g_sfx, session)

    if model_name is None:
        p2_models = list(Path(p2_model_root).glob("*.pth"))
        if not p2_models:
            msg = f"No model found in '{p2_model_root}'."
            raise FileNotFoundError(msg)
        if len(p2_models) == 1:
            p2_model = p2_models.pop()
        else:
            p2_model = Path(browse_files(initialdir=p2_model_root, filetypes="pth"))
        model_name = p2_model.name.removesuffix("_final.pth")
    else:
        model_name = model_name.removesuffix("_final.pth")
        p2_model = p2_model_root / f"{model_name}_final.pth"

    # Load hyperparameters from table
    hp_tab = get_vgg_performance_table(exclusive_gender_trials=exclusive_gender_trials)
    model_hps = hp_tab[hp_tab.model_name == model_name]
    decision_block = model_hps.dblock.to_numpy()[0]
    freeze_vgg_core = model_hps.freeze_vgg_core.to_numpy()[0]
    last_core_layer = model_hps.last_core_layer.to_numpy()[0]
    parallel_bridge = model_hps.parallel_bridge.to_numpy()[0]

    # Initialize model
    if "FrozenCore" in model_name:
        if pd.Timestamp(model_name.split("_")[0]) < pd.Timestamp("2023-04-03"):
            model = VGGFaceHumanjudgmentFrozenCoreOld(decision_block=decision_block, session=session)
        else:
            model = VGGFaceHumanjudgmentFrozenCore(
                decision_block=decision_block,
                last_core_layer=last_core_layer,
                parallel_bridge=parallel_bridge,
                session=session,
            )

    else:
        model = VGGFaceHumanjudgment(
            decision_block=decision_block,
            freeze_vgg_core=freeze_vgg_core,
            last_core_layer=last_core_layer,
            parallel_bridge=parallel_bridge,
            session=session,
        )
    model.name = model_name

    # Load model weights
    model_dict = torch.load(p2_model, map_location=lambda storage, loc: storage)  # noqa: ARG005
    model.load_state_dict(model_dict)

    if device is not None:
        return model.to(device).float()
    return model.float()


def get_vgg_performance_table(
    sort_by_acc: bool = True, hp_search: bool = False, exclusive_gender_trials: str | None = None
) -> pd.DataFrame:
    """
    Get the performance table for `VGGFace` models.

    :param sort_by_acc: Sort table by accuracy.
    :param hp_search: True: Use hyperparameter search table.
    :param exclusive_gender_trials: For models trained on exclusive gender trials ['female' OR 'male'], OR None.
    :return: VGGface performance table.
    """
    acc_cols = ["train_acc", "val_acc", "test_acc"]

    exclusive_gender_trials = check_exclusive_gender_trials(exclusive_gender_trials=exclusive_gender_trials)
    if isinstance(exclusive_gender_trials, str):
        p2_table = (
            paths.data.models.behave.hp_search.hp_table_gender
            if hp_search
            else paths.data.models.behave.hp_table_gender
        )
        p2_table = Path(p2_table.format(gender=exclusive_gender_trials))
    else:
        p2_table = Path(
            paths.data.models.behave.hp_search.hp_table if hp_search else paths.data.models.behave.hp_table
        )

    if p2_table.exists():
        hp_tab = pd.read_csv(p2_table)
        if sort_by_acc:
            hp_tab = hp_tab.sort_values(by="test_acc", ascending=False)
    else:
        # initialize table
        hp_tab = pd.DataFrame(
            columns=[
                "model_name",  # str
                "session",  # "2D" | "3D"
                "data_mode",  # "2d-original" | "3d-reconstructions" | "3d-perspectives"
                "freeze_vgg_core",  # bool
                "last_core_layer",  # str
                "parallel_bridge",  # bool
                "dblock",  # architecture of decision head
                "bs",  # batch size
                "epochs",  # int
                "lr",  # learning rate
                "seed",  # int
                "device",  # "cpu:i" | "cuda:i"
                "n_heads",  # number of heads (main: 100)
                "n_train",  # int
                "n_val",  # int
                "time_taken",  # pd.Timedelta
                *acc_cols,  # float (accuracy) * 3
            ]
        )
    return hp_tab


def draw_model(
    model: VGGFace | VGGcore | VGGFaceHumanjudgment | VGGFaceHumanjudgmentFrozenCore,
    output: torch.Tensor,
    keep: bool = False,
) -> None:
    """Draw the computational graph of a given `VGG` model variant."""
    dot_graph = make_dot(var=output, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)

    graph_name = f"graph_{type(model).__name__}"
    if hasattr(model, "session"):
        graph_name += f"_session-{model.session}"
    model_path = Path(paths.data.CACHE, graph_name)
    if hasattr(model, "freeze_vgg_core"):
        model_path = model_path.with_name(model_path.name + f"_frozen_core-{model.freeze_vgg_core}")
    if hasattr(model, "decision_block_mode"):
        model_path = model_path.with_name(model_path.name + f"_db-{model.decision_block_mode}")

    # Draw
    dot_graph.view(filename=model_path, cleanup=True)

    if not keep:
        cinput(string="\nPress any key to close & delete the graph file.\n", col="y")
        model_path.with_suffix(".pdf").unlink()


def model_summary(model: torch.nn, input_size: list, batch_size: int = -1, device: str = "cpu"):
    """Create a `Tensorflow`-like model summary."""
    summary(model=model, input_size=input_size, batch_size=batch_size, device=device)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
