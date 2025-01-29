# !/usr/bin/env python3
"""Extract attributes from face images."""

# %% Import
from __future__ import annotations

import os
import platform
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vedo
from PIL import Image
from ut.ils import ask_true_false, cprint, get_n_cols_and_rows

from facesim3d.configs import params, paths

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@cache
def get_cfd_code_table():
    """Load the `CFD` code table."""
    __cfd_code_tab = pd.read_excel(
        io=Path(paths.data.CFD, "CFD 3.0 Norming Data and Codebook.xlsx"),
        sheet_name="CFD 3.0 Codebook",
        header=11,
        usecols=range(6),
    )
    return __cfd_code_tab.drop(index=0)


def cfd_var_converter(var_id_or_label: str) -> pd.DataFrame:
    """Convert the `CFD` variable ID to a corresponding label name, and vice versa."""
    cfd_code_tab = get_cfd_code_table()

    if var_id_or_label[1].isnumeric():
        # ID -> label name
        var_return = cfd_code_tab.VarLabel[cfd_code_tab.VarID.str.contains(var_id_or_label, case=False)].item()
    else:
        # label name -> ID
        var_return = cfd_code_tab.VarID[cfd_code_tab.VarLabel.str.match(var_id_or_label, case=False)]

        if len(var_return) > 1:
            print("Note, there are various IDs with that label name:", var_return.to_list())

    return var_return


def cfd_var_description(var_id_or_label: str) -> pd.DataFrame:
    """Get the description for a `CFD` variable ID or label name."""
    cfd_code_tab = get_cfd_code_table()

    if var_id_or_label[1].isnumeric():
        # ID -> description
        var_return = cfd_code_tab.Description[cfd_code_tab.VarID.str.contains(var_id_or_label, case=False)].item()
    else:
        # label name -> description
        var_return = cfd_code_tab.Description[cfd_code_tab.VarLabel.str.match(var_id_or_label, case=False)]

        if len(var_return) > 1:
            print("Note, there are various descriptions with that label name:", var_return.to_list())

    return var_return


def get_cfd_features(
    set_name: str = "main", drop_nan_col: bool = True, physical_attr_only: bool = True
) -> pd.DataFrame:
    """
    Get `CFD` features.

    :param set_name: CFD set name
    :param drop_nan_col: whether to drop columns that contain only nan
    :param physical_attr_only: whether to drop columns that do not represent physical attributes
    :return: CFD feature table
    """
    set_name = set_name.lower()
    set_dict = {
        "main": "CFD U.S. Norming Data",
        "mr": "CFD-MR U.S. Norming Data",
        "india-us": "CFD-I U.S. Norming Data",
        "india-ind": "CFD-I INDIA Norming Data",
    }
    set_names = list(set_dict.keys())
    if set_name not in set_names:
        msg = f"set_name must be in {set_names}!"
        raise ValueError(msg)
    cfd_tab = pd.read_excel(
        io=Path(paths.data.CFD, "CFD 3.0 Norming Data and Codebook.xlsx"),
        sheet_name=set_dict[set_name],
        header=6,
        index_col=0,
    )
    cfd_tab = cfd_tab.drop(index=np.nan)
    cfd_tab = cfd_tab.drop(index="Model")
    cfd_tab.index = cfd_tab.index.set_names(names="Model")

    if drop_nan_col:
        for col in cfd_tab.columns:
            if (not cfd_tab[col].notna().any()) or (physical_attr_only and col[0] != "P"):
                # if all entries in column are nan OR it is not a physical attribute: drop column
                cfd_tab = cfd_tab.drop(columns=col)

    return cfd_tab


def get_cfd_features_for_models(
    list_of_models: list[Any] | None = None, physical_attr_only: bool = True
) -> pd.DataFrame:
    """
    Get `CFD` features for the given list of head models.

    :param list_of_models: list of models
    :param physical_attr_only: whether to drop columns that do not represent physical attributes
    :return: feature table for a list of models
    """
    _feat_tab = get_cfd_features(physical_attr_only=physical_attr_only)
    list_of_models = _feat_tab.index if list_of_models is None else list_of_models
    return _feat_tab.loc[list_of_models]


@cache
def get_head_mapper_table():
    """Get the table for mapping head numbers to head indices."""
    return pd.read_csv(Path(paths.data.unity.cfd, "headnmap.csv"), names=["CFD_filename", "idx", "unknown", "head_nr"])


def head_nr_to_pilot_matrix_index(head_id: str | int | None, pilot_version: int = 2) -> int:
    """
    Convert a head number to a corresponding matrix index for the pilot studies.

    Convert the head number (as they appear in the rating files, `'*TrialResults*.csv'`) to the pilot index
    as it appears, e.g., in the similarity matrix.

    :param head_id: Head number, as [str], e.g., "Head4", or as [int], e.g., 4.
    :param pilot_version: version of pilot (v1, v2)
    :return: head index
    """
    pv1, pv2 = 1, 2
    if pilot_version not in {pv1, pv2}:
        msg = "pilot_version must be 1 or 2!"
        raise ValueError(msg)

    n_fm = (12, 13) if pilot_version == pv2 else (15, 15)  # number of faces per gender (f, m)

    if isinstance(head_id, str):
        head_id = int(head_id.title().removeprefix("Head"))

    if head_id not in range(1, n_fm[0] + 1) and head_id not in range(51, 51 + n_fm[1]):
        msg = "head_id is out of range!"
        raise ValueError(msg)

    n_face_split = params.main.n_faces // 2
    head_idx = (
        head_id - 1 if head_id <= n_face_split else head_id - (n_face_split + 1) + n_fm[0]
    )  # female or male faces

    return int(head_idx)


def head_nr_to_main_matrix_index(head_id: str | int | None) -> int:
    """
    Convert a head number to a corresponding matrix index in the main study.

    Convert the head number (as they appear in the rating files, `'*TrialResults*.csv'`) to the index as it
    appears, e.g., in the similarity matrix.

    :param head_id: Head number, as [str], e.g., "Head4", or as [int], e.g., 4.
    :return: head index
    """
    if isinstance(head_id, str):
        head_id = int(head_id.title().removeprefix("Head"))
    if head_id not in range(1, params.main.n_faces + 1):
        msg = "head_id is out of range!"
        raise ValueError(msg)
    head_idx = head_id - 1
    return int(head_idx)


def pilot_matrix_index_to_head_nr(pilot_face_idx: int, pilot_version: int = 2) -> str:
    """
    Convert the matrix index of a head to a head number in the pilot studies.

    Convert the pilot index as it appears, e.g., in the similarity matrix to the corresponding head number
    (as they appear in the rating files, `'*TrialResults*.csv'`).

    !!! note
        This is the inverse function of `head_nr_to_pilot_matrix_index()`.

    :param pilot_face_idx: head index in the pilot matrix
    :param pilot_version: version of pilot experiment (v1, v2)
    :return: the head number
    """
    pv1, pv2 = 1, 2
    if pilot_version not in {pv1, pv2}:
        msg = f"pilot_version must be {pv1} or {pv2}!"
        raise ValueError(msg)

    n_fm = (12, 13) if pilot_version == pv2 else (15, 15)  # number of faces per gender (f, m)

    if pilot_face_idx not in range(n_fm[0] + n_fm[1]):
        msg = "pilot_face_idx is out of range!"
        raise ValueError(msg)

    head_id = pilot_face_idx + 1 if pilot_face_idx < n_fm[0] else pilot_face_idx + 51 - n_fm[0]  # female or male faces

    return f"Head{head_id}"


def main_matrix_index_to_head_nr(face_idx: int) -> str:
    """
    Convert the index to a corresponding head number in the main study.

    Convert the main index as it appears, e.g., in the similarity matrix to the corresponding head number
    (as they appear in the rating files, `'*TrialResults*.csv'`).

    !!! note
        This is the inverse function of `head_nr_to_main_matrix_index()`.

    :param face_idx: head index in the main study
    :return: the head number
    """
    if face_idx not in range(100):
        msg = "face_idx is out of range!"
        raise ValueError(msg)
    head_id = face_idx + 1
    return f"Head{head_id}"


def head_index_to_head_nr(face_idx: int) -> str:
    """
    Convert the head index (`'idx'` in `'headnmap.csv'`) to the head number (`'Head#'`).

    !!! note
        This is the inverse function of `head_nr_to_index()`.

    !!! note "For previous pilot experiments"
        This is not being used for the pilot experiment with fewer heads.

    :return: head number
    """
    map_tab = get_head_mapper_table()
    return map_tab.loc[map_tab.idx == face_idx, "head_nr"].values[0]  # 'Head#'  # noqa: PD011


def head_nr_to_index(head_id: str | int | None) -> int:
    """
    Convert the head number (`'Head#'`) to the head index (`'idx'` in `'headnmap.csv'`).

    :param head_id: Head number, as [str], e.g., "Head4", or as [int], e.g., 4.
    :return: head index
    """
    map_tab = get_head_mapper_table()
    if isinstance(head_id, int):
        head_id = f"Head{head_id}"
    return map_tab.loc[map_tab.head_nr == head_id, "idx"].values[0]  # noqa: PD011


def pilot_index_to_model_name(pilot_face_idx: int, pilot_version: int = 2) -> str:
    """
    Convert the head index to the head model name.

    Convert the pilot index as it appears, e.g., in the similarity matrix to the corresponding name of the head model
    (as it appears in the `CFD` feature table (`PFA`)).

    :param pilot_face_idx: head index in the pilot experiment
    :param pilot_version: the version of pilot experiment (v1, v2)
    :return: name of the head model
    """
    pv1, pv2 = 1, 2
    if pilot_version not in {pv1, pv2}:
        msg = "pilot_version must be 1 or 2!"
        raise ValueError(msg)

    n_fm = (12, 13) if pilot_version == pv2 else (15, 15)  # number of faces per gender (f, m)

    if pilot_face_idx not in range(n_fm[0] + n_fm[1]):
        msg = "pilot_face_idx is out of range!"
        raise ValueError(msg)

    head_nr = pilot_matrix_index_to_head_nr(pilot_face_idx=pilot_face_idx, pilot_version=pilot_version)
    convert_tab = get_head_mapper_table()
    model_name = convert_tab[convert_tab.head_nr == head_nr].CFD_filename.item()
    return model_name.removeprefix("CFD-")[:6]


def main_index_to_model_name(face_idx: int) -> str:
    """
    Convert the head index to a head model name in the main study.

    Convert the main study index as it appears, e.g., in the similarity matrix to the name of the corresponding head
    model (as it appears in the `CFD` feature table (`PFA`)).

    :param face_idx: head index in pilot
    :return: name of the head model
    """
    if face_idx not in range(100):
        msg = "face_idx is out of range!"
        raise ValueError(msg)

    head_nr = main_matrix_index_to_head_nr(face_idx=face_idx)
    convert_tab = get_head_mapper_table()
    model_name = convert_tab[convert_tab.head_nr == head_nr].CFD_filename.item()
    return model_name.removeprefix("CFD-")[:6]


def heads_naming_converter_table(pilot_version: int | None = None) -> pd.DataFrame:
    """
    Load the table for head naming conversion.

    :param pilot_version: None: for the main experiment, OR for pilot: 1, OR: 2.
    :return: converter table
    """
    pv1, pv2 = 1, 2

    # Load mapping table
    heads_tab = get_head_mapper_table()
    # Keep only samples where reconstructions are computed/used
    heads_tab = heads_tab[~heads_tab.head_nr.str.match("_")]
    # Keep only the necessary columns
    heads_tab = heads_tab.drop(columns=["idx", "unknown"], inplace=False).copy()

    if isinstance(pilot_version, int):
        # Extract pilot heads
        if pilot_version == pv1:
            heads_tab = heads_tab[heads_tab.head_nr.str.fullmatch(r"Head([1-9]|1[0-5]|5[1-9]|6[0-5])")]
        elif pilot_version == pv2:
            heads_tab = heads_tab[heads_tab.head_nr.str.fullmatch(r"Head([1-9]|1[0-2]|5[1-9]|6[0-3])")]
        else:
            msg = f"pilot_version must be None, {pv1}, OR {pv2}."
            raise ValueError(msg)

        # # Remove .jpg suffix

    # Extract and save naming of Model
    heads_tab["Model"] = [fn[4:10] for fn in heads_tab.CFD_filename]

    return heads_tab


def face_image_path(
    head_id: str | int,
    data_mode: str = "3d-reconstructions",
    return_head_id: bool = False,
    angle: float | str | None = None,
) -> str | tuple[str, int]:
    """
    Construct the path to a face image.

    :param head_id: Head number, as [str], e.g., "Head4", or as [int], e.g., 4.
    :param data_mode: path to the "2d-original", "3d-reconstructions", or "3d-perspectives"
    :param return_head_id: whether to return the head number (as it appears in the rating files, '*TrialResults*.csv')
    :param angle: for data_mode=="3d-perspectives", a face angle needs to be given.
    :return: path to the face image
    """
    data_mode = data_mode.lower()

    # Get mapping table
    head_map_tab = get_head_mapper_table()

    if isinstance(head_id, str):
        # Map ID to head index
        if head_id.startswith("CFD-"):
            head_id = head_map_tab[head_map_tab.CFD_filename == head_id + ".jpg"].idx.item()
        elif head_id.startswith("Head"):
            head_id = head_map_tab[head_map_tab.head_nr == head_id].idx.item()
        else:
            msg = f"{head_id = } unknown!"
            raise ValueError(msg)

    if data_mode == "3d-perspectives":
        if angle is None:
            msg = "For data_mode == '3d-perspectives' angle must be given [int | float | str == 'frontal']!"
            raise ValueError(msg)

        if isinstance(angle, str):
            angle = angle.lower()

        if angle == "frontal":
            path_to_image = Path(paths.data.cfd.faceviews, f"head-{head_id:03d}_frontal.png")

        else:
            # even for floats (e.g., 348.75), integer parts are all unique, so search for those
            angle = round(float(angle), 2)  # 348.7512 -> 348.75, 315 -> 315.00
            angle = int(angle) if angle.is_integer() else angle  # in case of int, map back: 315.00 -> 315
            path_to_image = Path(paths.data.cfd.faceviews, f"head-{head_id:03d}_angle-{angle}.png")

    else:
        th_gender = 60  # threshold
        path_to_pid = Path(paths.data.unity.cfd, "female" if head_id <= th_gender else "male", str(head_id))

        path_to_image = str(
            path_to_pid / f"{head_id}_inputs.jpg"
            if "original" in data_mode
            else path_to_pid / f"{head_id}_screenshot.png"
        )

    if return_head_id:
        return path_to_image, head_id
    return path_to_image


def display_face(
    head_id: str | int,
    data_mode: str = "3d-reconstructions",
    angle: str | float | None = None,
    interactive: bool = False,
    show: bool = True,
    verbose: bool = False,
) -> Image:
    """
    Display the `CFD` face given its head ID.

    :param head_id: either image name (e.g. "CFD-WF-001-003-N"), OR head number (e.g., "Head5"), OR
                    head index
    :param data_mode: path to the "2d-original", "3d-reconstructions", or "3d-perspectives"
    :param angle: viewing angle of face to display [only if data_mode == "3d-perspectives"]
    :param interactive: if True display 3D (.obj) in interactive mode.
    :param show: if True show image
    :param verbose: if True print the path to the image.
    """
    data_mode = data_mode.lower()
    if data_mode == "3d-perspectives" and angle is None:
        msg = "angle must be given for data_mode == '3d-perspectives'"
        raise ValueError(msg)

    original = "original" in data_mode or "3d-perspectives" in data_mode
    if original and interactive:
        cprint(string="For interactively displaying a 3D face mesh, set 'original' to False!", col="r")

    # Display image
    path_to_image, head_id = face_image_path(head_id=head_id, data_mode=data_mode, return_head_id=True, angle=angle)

    if original:
        face_image = Image.open(path_to_image)
        if show:
            face_image.show()

    else:
        path_to_3d_obj = path_to_image.replace(f"{head_id}_screenshot.png", f"{head_id}.obj")
        face_image = vedo.Mesh(inputobj=path_to_3d_obj)
        face_image.texture(tname=path_to_3d_obj.replace(".obj", ".png"), scale=0.1)

        if interactive:
            # This is buggy on Mac. Cannot be called multiple times in one session.
            # Documentation (incl. shortcuts): https://vedo.embl.es/autodocs/content/vedo/index.html
            cprint(string="\nPress 'q' to close window!", col="b")
            face_image.show().close()
        else:
            if not Path(path_to_image).is_file():
                vedo.settings.screenshotTransparentBackground = True
                plotter = vedo.Plotter(interactive=interactive, offscreen=True)
                plotter.show(face_image, zoom=1.8)
                plotter.screenshot(path_to_image, scale=1).close()
            face_image = Image.open(path_to_image)
            if show:
                face_image.show()

    if verbose:
        cprint(string=f"\nImage path of '{head_index_to_head_nr(face_idx=head_id)}': {path_to_image}", col="b")

    return face_image


def display_set_of_faces(
    list_head_ids: list[str | int], data_mode: str, num_suffix: str = "", verbose: bool = False
) -> tuple[Any, np.ndarray]:
    """Display a set of heads in a grid."""
    # Prepare data_mode
    data_mode = data_mode.lower()
    data_mode_suffix = (
        "original" if "original" in data_mode else "3D-reconstructed" if "3d-recon" in data_mode else "3D-perspectives"
    )

    # Load images
    list_of_imgs = []
    for head_id in list_head_ids:
        list_of_imgs.append(
            display_face(head_id=head_id, data_mode=data_mode, interactive=False, show=False, verbose=verbose)
        )

    # Plot in grid
    # TODO: use this function in display_representative_faces() in computational_choice_model.py  # noqa: FIX002
    grid_shape = get_n_cols_and_rows(n_plots=len(list_head_ids), square=True)
    fig, axes = plt.subplots(
        *grid_shape,
        figsize=(grid_shape[1] * 2, grid_shape[0] * 2),
        sharex=True,
        sharey=True,
        num=f"{len(list_head_ids)} of the {data_mode_suffix} CFD faces {num_suffix}",
    )

    for i, (face_id, face_img) in enumerate(zip(list_head_ids, list_of_imgs, strict=True)):
        axes.flatten()[i].imshow(face_img)
        axes.flatten()[i].set_xticks([])
        axes.flatten()[i].set_xlabel(face_id)
        axes.flatten()[i].yaxis.set_visible(False)
        for spine in axes.flatten()[i].spines.values():  # remove axes-box around image
            spine.set_visible(False)
        fig.tight_layout()
        plt.show()

    return fig, axes


def list_faulty_heads(run: bool = False, suffix: str = "") -> pd.DataFrame:
    """
    List faulty heads (i.e., heads for which the reconstruction is not optimal).

    DECA has a reported reconstructed error in the eyes (misalignment of the eyes), and open mouth,
    which is not open in the original image.

    :param run: if True: open all images and fill in the faulty heads in the table.
    :param suffix: path suffix hinting to the focus of the observation among the face stimuli (e.g., 'eyes')
    :return: the list of faulty heads (IDs)
    """
    # Set path to faulty heads table
    suffix = f"_{suffix}" if suffix else ""
    path_to_table = Path(paths.data.unity.cfd, f"faulty_heads{suffix}.csv")

    if run:
        if path_to_table.is_file():
            print(f"{path_to_table} exists already!")
            df_faulty_heads = pd.read_csv(path_to_table, index_col=0)
            if not ask_true_false("Do you want to add missing heads to table?"):
                return df_faulty_heads
            # In case all heads shall be replaced, delete table manually

        else:
            df_faulty_heads = pd.DataFrame(columns=["head_nr", "faulty"])
            df_faulty_heads = df_faulty_heads.set_index("head_nr")

        for i in range(1, 159 + 1):
            # TODO: ?  # noqa: FIX002
            head_nr = head_index_to_head_nr(face_idx=i)

            if head_nr in df_faulty_heads.index or head_nr == "_":
                continue

            try:
                display_face(head_id=head_nr, data_mode="3d-reconstructions", interactive=False, verbose=True)
                q = suffix or "eyes and/or mouth"
                df_faulty_heads.loc[head_nr, "faulty"] = int(ask_true_false(f"{head_nr}: Are/is {q} corrupted?"))
                if platform.system().lower() == "darwin":
                    os.system("""/usr/bin/osascript -e 'tell app "Preview" to close (first window)' """)  # noqa: S605
            except AttributeError:
                cprint(string=f"ID '{head_nr}' not valid!", col="r")

            # Save table
            df_faulty_heads.to_csv(path_to_table)

    else:
        # Load existing table
        df_faulty_heads = pd.read_csv(path_to_table, index_col=0)

    return df_faulty_heads


# TODO: implement class  # noqa: FIX002
class FaceViews:
    """Class for face views."""

    def __init__(self) -> None:
        """Initialise FaceViews class."""
        self.path_to_views = paths.data.cfd.faceviews
        self._check_path_to_views()

    def _check_path_to_views(self) -> None:
        """Check if the path to face views exists."""
        if not Path(self.path_to_views).is_dir():
            cprint(string=f"Path to face views does not exist: {self.path_to_views}", col="r")
            print(
                "Download face views from "
                "Keeper:'NEUROHUM/Material/Face Datasets/3D-reconstructions_different_angles/' and "
                "save them to the path above."
            )


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from facesim3d.modeling.compute_similarity import cosine_similarity

    Image.open(
        Path(paths.results.main.rdms, "Similarity (cosine) of CFD_PFF face features (90%-PCA)_rsatoolbox.png")
    ).show()  # check this out

    # Upper left triangle of the similarity matrix vs. rest in females
    display_set_of_faces(
        list_head_ids=list(map(main_matrix_index_to_head_nr, range(33))), data_mode="2d-original", verbose=False
    )
    display_set_of_faces(
        list_head_ids=list(map(main_matrix_index_to_head_nr, range(33, 50))), data_mode="2d-original", verbose=False
    )

    # Display some manually selected faces
    display_set_of_faces(
        list_head_ids=list(
            map(
                main_matrix_index_to_head_nr,
                [
                    2,  # very similar to most of the first 33 female heads
                    10,
                    13,  # very dissimilar, according to PFF-PCA below
                    18,  # as 14 different to most other female heads
                    39,
                    49,
                ],
            )
        ),  # very similar, according to the PFF-PCA below
        data_mode="2d-original",
        verbose=False,
    )

    # Get fitted PCA model
    head_map = heads_naming_converter_table(pilot_version=None)
    feat_tab = get_cfd_features_for_models(list_of_models=head_map.Model, physical_attr_only=True)

    # Scale data
    scaler = StandardScaler()
    scaled_feat_tab = scaler.fit_transform(feat_tab.to_numpy())

    pca_model = PCA(
        n_components=0.9,
        svd_solver="full",  # must be 0 < pca < 1 for svd_solver="full" (see docs)
    )  # .set_output(transform="pandas")
    # this finds n components such that pca*100 % of variance is explained
    pca_feat_tab = pca_model.fit_transform(scaled_feat_tab)  # (n_faces, n_features)

    print(f"{cosine_similarity(pca_feat_tab[10, :], pca_feat_tab[13, :]) =:.2f}")  # different heads
    print(f"{cosine_similarity(pca_feat_tab[39, :], pca_feat_tab[49, :]) =:.2f}")  # similar heads

    for i in range(pca_model.n_components_):
        var_explained = pca_model.explained_variance_ratio_[i]
        print(f"\nPC{i + 1} explains {var_explained:.1%} of variance.")
        if var_explained < 0.1:  # noqa: PLR2004
            continue

        feature_loading_pc = pca_model.components_[i, :]  # how much each feature contributes to this PC
        abs_feature_loading_pc = np.abs(feature_loading_pc)

        indices_sorted_by_importance = np.argsort(abs_feature_loading_pc)[::-1]  # strongest to weakest
        # idx_most_imp_feat = np.abs(pca_model.components_[i, :]).argmax()  # noqa: ERA001
        # == indices_sorted_by_importance[0]
        # feature_loading_pc[indices_sorted_by_importance]  # noqa: ERA001

        # Normalise absolute feature loading
        norm_abs_feature_loading_pc = abs_feature_loading_pc / abs_feature_loading_pc.sum()  # sums to 1

        # Take the most positive and most negative, and closest to zero value of the PC and display the
        # corresponding faces
        idx_most_representative_heads = np.abs(pca_feat_tab[:, i]).argsort()[::-1]
        most_pos_head_idx = pca_feat_tab[:, i].argmax()
        most_pos_head_nr = main_matrix_index_to_head_nr(most_pos_head_idx)
        most_neg_head_idx = pca_feat_tab[:, i].argmin()
        most_neg_head_nr = main_matrix_index_to_head_nr(most_neg_head_idx)
        close_zero_head_idx = idx_most_representative_heads[-1]
        close_zero_head_nr = main_matrix_index_to_head_nr(close_zero_head_idx)

        print(f"\t\tThe head with the most negative loading in PC{i + 1} is: '{most_neg_head_nr}'")
        print(f"\t\tThe head with the closest-to-zero loading in PC{i + 1} is: '{close_zero_head_nr}'")
        print(f"\t\tThe head with the most positive loading in PC{i + 1} is: '{most_pos_head_nr}'")

        fig, axes = display_set_of_faces(
            list_head_ids=[most_neg_head_nr, close_zero_head_nr, most_pos_head_nr],
            data_mode="2d-original",
            num_suffix=f"| PC{i + 1}",
            verbose=False,
        )

        # Add boxes around heads
        head_colors = ["blue", "black", "red"]
        for color, ax in zip(head_colors, axes.flatten(), strict=True):
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1)
                spine.set_visible(True)

        # Explore 3 most important features
        for j, idx_imp_feat in enumerate(indices_sorted_by_importance[:3]):
            feat_code = feat_tab.columns[idx_imp_feat]
            direction = "negative" if feature_loading_pc[idx_imp_feat] < 0 else "positive"
            # positive: greater values in this feature means greater importance in PC
            # negative: smaller values in this feature means greater importance in PC
            print(
                f"\tThe {j + 1}. most important {direction} feature "
                f"({norm_abs_feature_loading_pc[idx_imp_feat]:.1%} loading) in PC{i + 1} is: "
                f"'{cfd_var_converter(feat_code)}' ({feat_code})"
            )

            # Plot histogram of feature values
            plt.figure(figsize=(8, 6))
            ax = feat_tab.loc[:, feat_code].hist()
            for lab, color, hidx in zip(
                ["most neg", "closest zero", "most pos"],
                head_colors,
                [most_neg_head_idx, close_zero_head_idx, most_pos_head_idx],
                strict=True,
            ):
                ax.axvline(x=feat_tab.iloc[hidx][feat_code], color=color, label=lab)
            plt.legend()
            ax.set_title(f"{j + 1}. most important {direction} feature in PC{i + 1} | {cfd_var_converter(feat_code)}")

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
