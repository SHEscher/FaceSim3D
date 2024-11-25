"""Prepare data for `VGG` models."""

# %% Import
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from ut.ils import cprint, dims_to_rectangularize, rectangularize_1d_array

from facesim3d.configs import params, paths
from facesim3d.modeling.face_attribute_processing import display_face, face_image_path, head_nr_to_index
from facesim3d.modeling.VGG.models import get_vgg_layer_names
from facesim3d.read_data import read_trial_results_of_session

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
VGG_LAYER_NAMES = get_vgg_layer_names()


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def load_image_for_model(image_path: str | Path, dtype: torch.float64, subtract_mean: bool = True) -> torch.Tensor:
    """Load an image for the `VGG` model."""
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, dsize=(224, 224))
    image = torch.Tensor(image).permute(2, 0, 1).view(1, 3, 224, 224).to(dtype)
    if subtract_mean:
        # this subtraction should be the average pixel value of the training set of the original VGGFace
        image -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).to(dtype).view(1, 3, 1, 1)
    return image


def revert_model_image(image: torch.Tensor, add_mean: bool) -> np.ndarray:
    """Revert a model-input-image to its original form."""
    if add_mean:
        image += torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).to(image.dtype).view(1, 3, 1, 1)
    return image[0].permute(1, 2, 0).to("cpu").numpy().astype(np.uint8)


class VGGFaceHumanjudgmentDataset(Dataset):
    """Dataset for the `VGG-Face` model variant, adapted for human similarity judgments."""

    def __init__(
        self,
        session: str,
        frozen_core: bool,
        data_mode: str = "2d-original",
        last_core_layer: str | None = None,
        dtype: torch.dtype = torch.float32,
        size: int | None = None,
        exclusive_gender_trials: str | None = None,
        heads: list[int] | np.ndarray[int] | int | None = None,
        **kwargs,
    ) -> None:
        """Initialize the `VGGFaceHumanjudgmentDataset`."""
        self.session = session
        self._size = size
        self.exclusive_gender_trials = exclusive_gender_trials
        self._heads = heads
        self.session_data = read_trial_results_of_session(session=session, clean_trials=True, verbose=False)[
            ["head1", "head2", "head3", "head_odd"]
        ].astype(int)
        self.frozen_core = frozen_core
        self._vgg_core_output = None
        self.data_mode = data_mode.lower()
        self.last_core_layer = last_core_layer
        self._suffix_data_mode = (
            "original" if "orig" in self.data_mode else "3D-recon" if "recon" in self.data_mode else "3D-persp"
        )
        self.dtype = dtype
        self._subtract_mean = kwargs.pop("subtract_mean", True)
        self._current_index = None

    @property
    def session_data(self):
        """Return the session data."""
        return self._session_data

    @session_data.setter
    def session_data(self, value: pd.DataFrame):
        """Set the session data."""
        # Data checks
        if not isinstance(value, pd.DataFrame):
            msg = f"Session data must be a pandas DataFrame, not {type(value)}."
            raise TypeError(msg)

        if set(value.columns) != {"head1", "head2", "head3", "head_odd"}:
            msg = "Session data must have columns 'head1', 'head2', 'head3', 'head_odd'."
            raise ValueError(msg)

        # Set session data
        self._session_data = value

        # If required, take exclusive gender trials only
        if self.exclusive_gender_trials in params.GENDERS:
            gender_cut = params.main.n_faces // 2  # == 50 for the main experiment

            if self.exclusive_gender_trials == "female":
                self._session_data = self._session_data[(self._session_data <= gender_cut).all(axis=1)]
            else:
                self._session_data = self._session_data[(self._session_data > gender_cut).all(axis=1)]
            self._session_data = self._session_data.reset_index(drop=True)

        # If required, take a subset of data with specific heads
        if isinstance(self._heads, int):
            self._heads = np.random.choice(
                np.unique(self.session_data.to_numpy().flatten()), self._heads, replace=False
            )
        if isinstance(self._heads, list | np.ndarray):
            self._session_data = self.session_data.loc[self.session_data.isin(self._heads).all(axis=1)].reset_index(
                drop=True
            )

        # Take random subsample of size self._size
        if isinstance(self._size, int):
            if self._size > len(self.session_data):
                cprint(
                    string="The sample size can't be bigger than the actual dataset size. "
                    f"Size will be set to length of session data (={len(self)}).",
                    col="y",
                )
                self._size = min(self._size, len(self))
            else:
                self._session_data = self.session_data.sample(n=self._size, replace=False).reset_index(drop=True)

    @property
    def data_mode(self):
        """Return the data mode."""
        return self._data_mode

    @data_mode.setter
    def data_mode(self, value: str):
        """Set the data mode."""
        if not isinstance(value, str):
            msg = f"Data mode must be a string, not {type(value)}."
            raise TypeError(msg)
        value = value.lower()
        if value not in params.DATA_MODES:
            msg = f"Data mode must be in {params.DATA_MODES}."
            raise ValueError(msg)
        self._data_mode = value

    @property
    def last_core_layer(self):
        """Return the cut layer of the `VGG core` model."""
        return self._last_core_layer

    @last_core_layer.setter
    def last_core_layer(self, layer: str | None):
        """Set the core cut layer."""
        if layer is None and self.frozen_core:
            msg = "If frozen_core is True, last_core_layer must be given!"
            raise ValueError(msg)
        if layer is None:
            pass
        elif isinstance(layer, str):
            layer = layer.lower()
            if layer not in VGG_LAYER_NAMES:
                msg = f"Core cut layer must be in {VGG_LAYER_NAMES}."
                raise ValueError(msg)
            if "-dropout" in layer:
                # Replace dropout layer with the previous relu layer
                layer = layer.replace("-dropout", "-relu")
                cprint(
                    string="Getting the data of the dropout layer is not possible!\n"
                    f"Instead, the data will be taken from the previous layer: '{layer}'",
                    col="y",
                )

        else:
            msg = f"Core cut layer must be a string or None, not {type(layer)}."
            raise TypeError(msg)
        self._last_core_layer = layer

    @property
    def exclusive_gender_trials(self) -> str | None:
        """Return the `exclusive_gender_trials` configuration."""
        return self._exclusive_gender_trials

    @exclusive_gender_trials.setter
    def exclusive_gender_trials(self, value: str | None) -> None:
        """Set the `exclusive_gender_trials`."""
        if value is not None:
            msg = f"exclusive_gender_trials must be in {params.GENDERS} OR None."
            if isinstance(value, str):
                value = value.lower()
                if value not in params.GENDERS:
                    raise ValueError(msg)
            else:
                raise TypeError(msg)
        self._exclusive_gender_trials = value

    @property
    def current_index(self):
        """Return the current index."""
        return self._current_index

    @property
    def vgg_core_output(self):
        """Return the `VGG core` output."""
        if self._vgg_core_output is None:
            # Set path to feature map
            #   take the output of max pool after conv_5_3 layer, since it showed the highest R with
            #   human judgments in the RSA
            ln_out = self.last_core_layer  # former output came from "fc7-relu" layer [before 2023-04-03]
            p2_feat_map = Path(
                paths.results.heads.vggface,
                f"VGGface_feature_maps_{self._suffix_data_mode}_{ln_out}.pd.pickle",
            )
            self._vgg_core_output = pd.read_pickle(p2_feat_map).to_numpy()

        return self._vgg_core_output

    def display_triplet(self, idx: int | torch.Tensor, as_seen_by_model: bool = True, verbose: bool = False) -> None:
        """Display a triplet of images."""
        if torch.is_tensor(idx):
            idx = idx.tolist().pop()

        img1, img2, img3, _, _ = self[idx].values()  # _, _ == choice, idx
        faces_imgs = [img1, img2, img3]
        min_val = min(img.min() for img in faces_imgs)
        max_val = max(img.max() for img in faces_imgs)
        h1, h2, h3, odd = self.session_data.iloc[idx].to_numpy()
        faces = [f"Head{h}" for h in [h1, h2, h3]]

        choice_side = [h1, h2, h3].index(odd)

        # Display faces
        title = f"Session: {self.session} | {self.data_mode} | as seen by model: {as_seen_by_model} | idx: {idx}"
        color = "darkorange" if self.frozen_core and as_seen_by_model else "royalblue"

        r, c = 12, 4
        if self.frozen_core:
            x, y = dims_to_rectangularize(len(img1))
            c = round(r / x * y) - 1

        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, num=title, figsize=(r, c))
        for i, ax in enumerate(axs.flatten()):
            if as_seen_by_model:
                if self.frozen_core:
                    img = rectangularize_1d_array(arr=faces_imgs[i], wide=False)
                    ax.imshow(img, cmap="seismic", vmin=min_val, vmax=max_val)

                else:
                    img = faces_imgs[i].permute(1, 2, 0).to("cpu").numpy()
                    img = (img - img.min()) / (img.max() - img.min()) * 255
                    img = img.astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(Image.fromarray(img))
            else:
                ax.imshow(
                    display_face(
                        head_id=faces[i], data_mode=self.data_mode, interactive=False, show=False, verbose=verbose
                    )
                )
            ax.set_title(faces[i], color=color if i == choice_side else "black")
            if i == choice_side:
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.show()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.session_data)  # np.math.comb(params.main.n_faces, 3)

    def __getitem__(self, idx: int | torch.Tensor) -> dict[str, torch.Tensor]:
        """Get an item from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist().pop()

        self._current_index = idx

        h1, h2, h3, odd = self.session_data.iloc[idx].values  # noqa: PD011

        if self.frozen_core:
            inp_1 = torch.Tensor(self.vgg_core_output[h1 - 1, :].astype(float)).to(self.dtype)
            inp_2 = torch.Tensor(self.vgg_core_output[h2 - 1, :].astype(float)).to(self.dtype)
            inp_3 = torch.Tensor(self.vgg_core_output[h3 - 1, :].astype(float)).to(self.dtype)
        else:
            inp_1 = load_image_for_model(
                face_image_path(head_id=f"Head{h1}", data_mode=self.data_mode),
                dtype=self.dtype,
                subtract_mean=self._subtract_mean,
            ).squeeze()
            inp_2 = load_image_for_model(
                face_image_path(head_id=f"Head{h2}", data_mode=self.data_mode),
                dtype=self.dtype,
                subtract_mean=self._subtract_mean,
            ).squeeze()
            inp_3 = load_image_for_model(
                face_image_path(head_id=f"Head{h3}", data_mode=self.data_mode),
                dtype=self.dtype,
                subtract_mean=self._subtract_mean,
            ).squeeze()

        choice_idx = torch.tensor([h1, h2, h3].index(odd), dtype=torch.int64)  # int necessary for x-entropy

        # Convert chosen odd-one-out into one-hot encoding # not needed for CrossEntropy Loss
        return {"image1": inp_1, "image2": inp_2, "image3": inp_3, "choice": choice_idx, "index": idx}


def prepare_data_for_human_judgment_model(
    session: str,
    frozen_core: bool,
    data_mode: str,
    last_core_layer: str | None = None,
    split_ratio: tuple = (0.7, 0.15, 0.15),
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    dtype: torch.dtype = torch.float32,
    size: int | None = None,
    exclusive_gender_trials: str | None = None,
    heads: list[int] | np.ndarray[int] | int | None = None,
    **kwargs,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data for the `VGG-Face`-model for human similarity judgments.

    Split the data into a train, validation and test set.

    :param session: '2D' OR '3D'
    :param frozen_core: prepare data for frozen VGG core or not
    :param data_mode: use "2d-original", "3d-reconstructions", or "3d-perspectives" as input images
    :param last_core_layer: must be given if frozen_core is True
    :param split_ratio: ratio of train, validation and test set
    :param batch_size: batch size for dataloader
    :param shuffle: shuffle data
    :param num_workers: number of workers for dataloader
    :param dtype: data type for images
    :param size: optionally define total size of data (which then gets split)
    :param exclusive_gender_trials: use exclusive gender trials ['female' OR 'male'], OR None for all samples.
    :param heads: optionally define subset of data, provide a list of head IDs or total number of heads IDs
    :return: train_dataloader, validation_dataloader, test_dataloader
    """
    # Load all data
    all_data = VGGFaceHumanjudgmentDataset(
        session=session,
        frozen_core=frozen_core,
        data_mode=data_mode,
        last_core_layer=last_core_layer,
        dtype=dtype,
        size=size,
        exclusive_gender_trials=exclusive_gender_trials,
        heads=heads,
        **kwargs,
    )

    # Split into train, validation and test set
    if sum(split_ratio) != 1.0:
        msg = "Split ratio must sum up to 1."
        raise ValueError(msg)
    cprint(
        string="Splitting data into {:.0%} training, {:.0%} validation & {:.0%} test set ...".format(*split_ratio),
        col="b",
    )
    training_size = int(split_ratio[0] * len(all_data))
    validation_size = int(split_ratio[1] * len(all_data))
    test_size = len(all_data) - training_size - validation_size
    train_data, val_data, test_data = random_split(
        dataset=all_data, lengths=[training_size, validation_size, test_size]
    )

    # Create dataloaders
    train_dataloader = (
        DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if training_size > 0
        else None
    )
    validation_dataloader = (
        DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if validation_size > 0
        else None
    )
    test_dataloader = (
        DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if test_size > 0
        else None
    )

    return train_dataloader, validation_dataloader, test_dataloader


@cache
def get_multi_view_data() -> pd.DataFrame:
    """Get the multi-view data."""
    mv_tab = pd.DataFrame(columns=["head_nr", "head_idx", "angle", "img_path"])
    for head_nr in range(1, params.main.n_faces + 1):
        head_idx = head_nr_to_index(head_id=head_nr)
        for path_to_image in Path(paths.data.cfd.faceviews).glob(f"head-{head_idx:03d}_*.png"):
            angle = path_to_image.stem.split("_")[1].split("angle-")[-1]
            mv_tab.loc[len(mv_tab)] = (head_nr, head_idx, angle, path_to_image)

    # Sort table by head_nr and angle
    def adapt_angle(_angle: str) -> str:
        """Transform a given angle to a sortable str object."""
        if _angle != "frontal":
            _angle = f"{float(_angle):06.2f}"
        return _angle

    mv_tab["angle_sortable"] = mv_tab.angle.map(adapt_angle)
    mv_tab = mv_tab.sort_values(by=["head_nr", "angle_sortable"], ascending=[True, True])
    return mv_tab.reset_index(drop=True)


class VGGMultiViewDataset(Dataset):
    """Dataset for the `VGG-Multi-View-Face` model."""

    def __init__(
        self,
        frozen_core: bool,
        last_core_layer: str | None = None,
        dtype: torch.dtype = torch.float32,
        heads: list[int] | np.ndarray[int] | int | None = None,
        **kwargs,
    ) -> None:
        """Initialize the `VGGMultiViewDataset`."""
        self._heads = heads
        self.multi_view_data = get_multi_view_data()
        self.frozen_core = frozen_core
        self._vgg_core_output = None
        self.last_core_layer = last_core_layer
        self._suffix_data_mode = "3D-persp"
        self.dtype = dtype
        self._subtract_mean = kwargs.pop("subtract_mean", True)
        self._current_index = None

    @property
    def n_unique_faces(self):
        """Return the number of unique faces."""
        return self.multi_view_data.head_nr.nunique()

    @property
    def multi_view_data(self):
        """Return the session data."""
        return self._multi_view_data

    @multi_view_data.setter
    def multi_view_data(self, value: pd.DataFrame):
        """Set the multi-view data."""
        # Data checks
        if not isinstance(value, pd.DataFrame):
            msg = f"Multi-view data must be a pandas DataFrame, not {type(value)}."
            raise TypeError(msg)

        valid_columns = {"head_nr", "head_idx", "angle", "img_path", "angle_sortable"}
        if set(value.columns) != valid_columns:
            msg = f"Multi-view data must have columns {valid_columns}."
            raise ValueError(msg)

        # Set multi_view_data
        self._multi_view_data = value

        # If required, take a subset of data with specific heads
        if isinstance(self._heads, int):
            self._heads = np.random.choice(self._multi_view_data.head_nr.unique(), self._heads, replace=False)

        if isinstance(self._heads, list | np.ndarray):
            self._multi_view_data = self.multi_view_data.loc[
                self.multi_view_data.head_nr.isin(self._heads)
            ].reset_index(drop=True)

    @property
    def last_core_layer(self):
        """Return cut layer of the `VGG core` model."""
        return self._last_core_layer

    @last_core_layer.setter
    def last_core_layer(self, layer: str | None):
        """Set the cut layer of the `VGG core` model."""
        if layer is None and self.frozen_core:
            msg = "If frozen_core is True, last_core_layer must be given!"
            raise ValueError(msg)
        if layer is None:
            pass
        elif isinstance(layer, str):
            layer = layer.lower()
            if layer not in VGG_LAYER_NAMES:
                msg = f"Core cut layer must be in {VGG_LAYER_NAMES}."
                raise ValueError(msg)
            if "-dropout" in layer:
                # Replace dropout layer with the previous relu layer
                layer = layer.replace("-dropout", "-relu")
                cprint(
                    string="Getting the data of the dropout layer is not possible!\n"
                    f"Instead, the data will be taken from the previous layer: '{layer}'",
                    col="y",
                )

        else:
            msg = f"Core cut layer must be a string or None, not {type(layer)}."
            raise TypeError(msg)
        self._last_core_layer = layer

    @property
    def current_index(self):
        """Return the current index."""
        return self._current_index

    @property
    def vgg_core_output(self):
        """Return the `VGG core` output."""
        raise NotImplementedError
        if self._vgg_core_output is None:
            # Set path to feature map
            #   take the output of max pool after conv_5_3 layer, since it showed the highest R with
            #   human judgments in the RSA
            ln_out = self.last_core_layer  # former output came from "fc7-relu" layer [before 2023-04-03]
            # TODO: adapt to specific face angles  # noqa: FIX002
            p2_feat_map = Path(
                paths.results.heads.vggface,
                f"VGGface_feature_maps_{self._suffix_data_mode}_{ln_out}.pd.pickle",
            )
            self._vgg_core_output = pd.read_pickle(p2_feat_map).to_numpy()
        return self._vgg_core_output

    def display_image(self, idx: int | torch.Tensor, as_seen_by_model: bool = True, verbose: bool = False) -> None:
        """Display face images with a specific angle."""
        if torch.is_tensor(idx):
            idx = idx.tolist().pop()

        face_img, head_nr, angle, _ = self[idx].values()  # _ == idx
        face = f"Head{head_nr}"

        # Display faces
        title = f"3d-perspectives | angle: {angle}° | as seen by model: {as_seen_by_model} | idx: {idx}"

        r, c = 10, 8
        if self.frozen_core:
            x, y = dims_to_rectangularize(len(face_img))
            c = round(r / x * y) - 1

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, num=title, figsize=(r, c))

        if as_seen_by_model:
            if self.frozen_core:
                img = rectangularize_1d_array(arr=face_img, wide=False)
                ax.imshow(img, cmap="seismic", vmin=face_img.min(), vmax=face_img.max())

            else:
                img = face_img.permute(1, 2, 0).to("cpu").numpy()
                img = (img - img.min()) / (img.max() - img.min()) * 255
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(Image.fromarray(img))
        else:
            ax.imshow(
                display_face(
                    head_id=face,
                    data_mode="3d-perspectives",
                    angle=angle,
                    interactive=False,
                    show=False,
                    verbose=verbose,
                )
            )
        ax.set_title(face, color="black")
        ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.show()

    def __len__(self) -> int:
        """Return the length of dataset."""
        return len(self.multi_view_data)

    def __getitem__(self, idx: int | torch.Tensor) -> dict[str, torch.Tensor | int | Any]:
        """Get an item from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist().pop()

        self._current_index = idx

        head_nr, _, angle, _, _ = self.multi_view_data.iloc[idx].to_numpy()  # head_idx, img_path, angle_sortable == _

        if self.frozen_core:
            # TODO: adapt code below for vgg_core_output # noqa: FIX002
            inp = torch.Tensor(self.vgg_core_output[head_nr - 1, :].astype(float)).to(self.dtype)
        else:
            inp = load_image_for_model(
                image_path=face_image_path(
                    head_id=f"Head{head_nr}", data_mode="3d-perspectives", return_head_id=False, angle=angle
                ),
                dtype=self.dtype,
                subtract_mean=self._subtract_mean,
            ).squeeze()

        return {"image": inp, "head_nr": head_nr, "angle": angle, "index": idx}


def prepare_data_for_multi_view_model(
    frozen_core: bool,
    last_core_layer: str | None = None,
    split_ratio: tuple = (0.8, 0.2, 0.0),
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    dtype: torch.dtype = torch.float32,
    heads: list[int] | np.ndarray[int] | int | None = None,
    **kwargs,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data for the multi-view model.

    Split the data into a train, validation and test set.

    :param frozen_core: prepare data for frozen VGG core or not
    :param last_core_layer: must be given if frozen_core is True
    :param split_ratio: ratio of train, validation and test set.
                        The test set always contains the frontal views of faces.
                        If > (..., ..., 0.) take also more views into the test set.
    :param batch_size: batch size for dataloader
    :param shuffle: shuffle data
    :param num_workers: number of workers for the dataloader
    :param dtype: data type for images
    :param heads: optionally define subset of data, provide a list of head IDs or total number of heads IDs
    :return: train_dataloader, validation_dataloader, test_dataloader
    """
    # Load all data
    all_data = VGGMultiViewDataset(
        frozen_core=frozen_core,
        last_core_layer=last_core_layer,
        dtype=dtype,
        heads=heads,
        **kwargs,
    )

    # Split into train, validation and test set
    if not np.allclose(sum(split_ratio), 1):
        msg = "Split ratio must sum up to 1."
        raise ValueError(msg)

    if max(split_ratio) > 1.0:
        msg = "split_ratio elements must be in in [0., 1]."
        raise ValueError(msg)

    cprint(
        string="Splitting data into {:.0%} training, {:.0%} validation & {:.0%} test set ...".format(*split_ratio),
        col="b",
    )

    # Splits are done within face ID's. At least all frontal views are in the test set.
    n_images_per_face = all_data.multi_view_data.head_nr.value_counts().max()  # == 33 (including frontal view)
    training_size_per_face = int(split_ratio[0] * (n_images_per_face - 1))  # -1 for the frontal view
    if split_ratio[2] == 0.0:
        validation_size_per_face = n_images_per_face - 1 - training_size_per_face
    else:
        validation_size_per_face = int(split_ratio[1] * (n_images_per_face - 1))
    test_size_per_face = n_images_per_face - training_size_per_face - validation_size_per_face

    training_indices = []
    validation_indices = []
    test_indices = all_data.multi_view_data[all_data.multi_view_data.angle == "frontal"].index.tolist()
    for _head_nr, data_head_nr in all_data.multi_view_data.groupby("head_nr"):
        test_indices_for_head = data_head_nr[data_head_nr.angle == "frontal"].index.tolist()
        train_indices_for_head = (
            data_head_nr[data_head_nr.angle != "frontal"].sample(training_size_per_face, replace=False).index.tolist()
        )
        training_indices += train_indices_for_head
        val_indices_for_head = (
            data_head_nr.drop(index=train_indices_for_head + test_indices_for_head)
            .sample(validation_size_per_face, replace=False)
            .index.tolist()
        )
        validation_indices += val_indices_for_head

        test_indices_for_head += (
            data_head_nr.drop(index=train_indices_for_head + val_indices_for_head + test_indices_for_head)
            .sample(test_size_per_face - 1, replace=False)
            .index.tolist()
        )
        # Add test_indices_for_head to test_indices if not already in there
        test_indices += [idx for idx in test_indices_for_head if idx not in test_indices]

    assert set(training_indices) & set(validation_indices) & set(test_indices) == set()  # noqa: S101
    assert len(training_indices) + len(validation_indices) + len(test_indices) == len(all_data)  # noqa: S101
    assert len(training_indices) + len(validation_indices) + len(test_indices) == len(all_data)  # noqa: S101
    assert (all_data.multi_view_data.iloc[training_indices + validation_indices].angle != "frontal").all()  # noqa: S101

    # Create the subsets
    train_data, val_data, test_data = (
        Subset(all_data, indices) for indices in (training_indices, validation_indices, test_indices)
    )

    # Create dataloaders
    train_dataloader = (
        DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if len(training_indices) > 0
        else None
    )
    validation_dataloader = (
        DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if len(validation_indices) > 0
        else None
    )
    test_dataloader = (
        DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if len(test_indices) > 0
        else None
    )

    return train_dataloader, validation_dataloader, test_dataloader


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


if __name__ == "__main__":
    for test_heads in [None, 15, [4, 64, 100]][0:1]:  # TODO: remove slice to run over all # noqa: FIX002
        multi_view_dataset = VGGMultiViewDataset(
            frozen_core=False,  # TODO: test: True, False # noqa: FIX002
            last_core_layer="fc7-relu",  # TODO: test several layers # noqa: FIX002
            heads=test_heads,
        )

    print(multi_view_dataset.multi_view_data)
    temp_img, temp_head_nr, temp_angle, temp_idx = multi_view_dataset[10].values()
    print(type(temp_img), temp_img.shape)
    print(f"Head-Nr: {temp_head_nr}, angle = {temp_angle}°")
    multi_view_dataset.display_image(idx=temp_idx, as_seen_by_model=True)
    multi_view_dataset.display_image(idx=temp_idx, as_seen_by_model=False)

    # Test the following
    for sr in ((0.8, 0.2, 0.0), (0.7, 0.15, 0.15)):
        train_dl, validation_dl, test_dl = prepare_data_for_multi_view_model(
            frozen_core=False,  # TODO: test: True, False # noqa: FIX002
            last_core_layer="fc7-relu",
            split_ratio=sr,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            dtype=torch.float32,
            heads=None,
        )
        # Do some tests on train_dl, validation_dl, test_dl
        for data in train_dl:
            _img, _head_nr, _angle, _idx = data.values()
            print(
                "img.shape",
                _img.shape,
                "head_nr:",
                _head_nr.cpu().numpy(),
                "angle:",
                _angle,
                "idx:",
                _idx.cpu().numpy(),
            )
            assert (multi_view_dataset.multi_view_data.iloc[_idx.cpu().numpy()].angle == _angle).all()  # noqa: S101
            break

        for i, data in enumerate(test_dl):
            _img, _head_nr, _angle, _idx = data.values()
            print(
                "img.shape",
                _img.shape,
                "head_nr:",
                _head_nr.cpu().numpy(),
                "angle:",
                _angle,
                "idx:",
                _idx.cpu().numpy(),
            )
            assert (multi_view_dataset.multi_view_data.iloc[_idx.cpu().numpy()].angle == _angle).all()  # noqa: S101
            if _angle[0] != "frontal":
                print("Is not frontal view!")
            else:
                print("Is frontal view!")
            if i > 10:  # noqa: PLR2004
                break
        print(test_dl.dataset.dataset.multi_view_data.loc[test_dl.dataset.indices].angle.value_counts())
        pass


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
