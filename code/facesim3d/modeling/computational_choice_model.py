# !/usr/bin/env python3
"""
Computational models of human choice behavior.

The `SPoSE` model is inspired by Hebart et al. (Nat. Hum Beh., 2020), see Fig.1c.
The `VICE` model is taken from Muttenthaler et al. (arXiv, 2022)

??? quote "VICE or SPoSE"
    'VICE rivals or outperforms its predecessor, SPoSE, at predicting human behavior in the odd-one-out
    triplet task.
    Furthermore, VICE's object representations are more reproducible and consistent across random initializations.'
    - Muttenthaler et al. (arXiv, 2022)

Other computational embedding models are in `facesim3d.modeling.VGG`.
"""

# %% Import
from __future__ import annotations

import json
import logging
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from ut.ils import cprint, normalize

from facesim3d.configs import params, paths, update_logger_configs
from facesim3d.modeling.compute_noise_ceiling import get_mea_table
from facesim3d.modeling.compute_similarity import compute_pearson_correlation_between_two_feature_matrices
from facesim3d.modeling.face_attribute_processing import (
    cfd_var_converter,
    display_face,
    face_image_path,
    get_cfd_features_for_models,
    head_nr_to_main_matrix_index,
    head_nr_to_pilot_matrix_index,
    main_index_to_model_name,
    main_matrix_index_to_head_nr,
    pilot_index_to_model_name,
    pilot_matrix_index_to_head_nr,
)
from facesim3d.modeling.prep_computational_choice_model import (
    BEST_HP_SPOSE,
    compute_vice_similarity_matrix,
    create_path_from_vice_params,
    get_best_hp_vice,
    list_spose_model_performances,
    load_spose_weights,
    load_vice_weights,
)
from facesim3d.modeling.rsa import (
    SPEARMAN,
    pearsonr,
    spearmanr,
    vectorize_similarity_matrix,
    visualise_matrix,
)
from facesim3d.read_data import read_pilot_data, read_pilot_participant_data, read_trial_results_of_session

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Number of faces
if params.PILOT:
    n_faces = params.pilot.v2.n_faces if params.PILOT_VERSION == 2 else params.pilot.v1.n_faces  # noqa: PLR2004
else:
    n_faces = params.main.n_faces

# VICE/SPoSE analysis
N_REPR_FACES: int = 3  # number of faces per dimension
M_DIMENSIONS: int = 6 if params.PILOT else 15  # OR 30 (for main, see init_dim below)
CORR_TH: float = 0.3
CHANCE_LVL: float = 1 / 3  # chance level for the triplet-odd-one-out task

# Hyperparameter search
HP_SEARCH: bool = False  # is done already

# Select dimensions
#   SPoSE
SPOSE_DIM_IND_2D: list = [0, 1, 2, 3, 6]  # all: []
SPOSE_DIM_IND_3D: list = [0, 2, 3, 7]  # all: []
#   VICE
VICE_DIM_IND_2D: list = list(range(5))  # all: []
VICE_DIM_IND_3D: list = list(range(5))  # all: []

# Logger
OVERWRITE_LOGGER = True  # overwrite logger file, when re-running the script

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# For sparse embedding models
def prepare_data_for_spose_and_vice(
    session: str, percentage: int | None = None, gender: bool | str = False, pilot: bool = params.PILOT
) -> None:
    """
    Prepare data for `SPoSE` & `VICE` models.

    !!! quote
        (...) triplets are expected to be in the format N x 3, where N = number of trials (e.g., 100k) and
        3 refers to the triplets, where col_0 = anchor_1, col_1 = anchor_2, col_2 = odd one out.
        Triplet data must be split into train and test splits, and named `train_90.txt` and `test_10.txt`,
        respectively.

    For hyperparameter tuning, prepare only a percentage of the data.

    For more information, see the repos of:
        [SPoSE](https://github.com/ViCCo-Group/SPoSE) & [VICE](https://github.com/LukasMut/VICE).

    :param session: '2D', OR '3D'
    :param percentage: percentage of data to use (e.g., 10, 20, ...)
    :param gender: True: use only triplets of the same gender, respectively. Compute recursively for both genders.
                   OR str: specify the gender 'female' or 'male'.
    :param pilot: True: use pilot data
    """
    # Check for gender specification
    if gender is True:
        prepare_data_for_spose_and_vice(session=session, percentage=percentage, gender="female", pilot=pilot)
        prepare_data_for_spose_and_vice(session=session, percentage=percentage, gender="male", pilot=pilot)
        return  # stop here

    if isinstance(gender, str):
        gender = gender.lower()
        if gender not in {"female", "male"}:
            msg = "Gender must be 'female' OR 'male'!"
            raise ValueError(msg)

    # Set paths
    spose_data_dir = Path(paths.data.pilot.v2, "for_SPoSE", session) if pilot else Path(paths.data.main.spose, session)
    if gender:
        spose_data_dir /= gender
    if percentage is not None:
        if percentage not in {10, 20, 30, 40, 50}:
            msg = "'percentage' must be in [10, 20, 30, 40, 50]!"
            raise ValueError(msg)
        spose_data_dir /= f"{percentage}perc"
    spose_data_dir.mkdir(parents=True, exist_ok=True)

    p2_training_set = spose_data_dir / "train_90.txt"
    p2_test_set = spose_data_dir / "test_10.txt"

    # Check if data already exists
    if p2_training_set.is_file() and p2_test_set.is_file():
        cprint(string=f"SPoSE & VICE data for {session} already prepared.", col="g")
        return

    # Load data tables
    if pilot:
        data_table = read_pilot_data(clean_trials=True, verbose=False)
        participant_table = read_pilot_participant_data()[["ppid", "group_exp"]]
        # Use data of one session (2D, 3D) only
        participant_table = participant_table.loc[participant_table.group_exp == session]
        data_table = data_table.loc[data_table.ppid.isin(participant_table.ppid)].reset_index(drop=True)
    else:
        data_table = read_trial_results_of_session(session=session, clean_trials=True, verbose=False)

    # Prepare training tables
    data_table = data_table[["head1", "head2", "head3", "head_odd"]]
    data_table = data_table.drop(
        index=data_table.loc[data_table.head_odd == 0].index, axis=1
    )  # remove trials w/o judgment
    data_table = data_table.dropna()

    # In the case of gender specification, filter data for gender-specific triplets
    gender_cut = params.main.n_faces // 2  # == 50
    if gender:
        if gender == "female":
            data_table = data_table[(data_table <= gender_cut).all(axis=1)]
        else:
            data_table = data_table[(data_table > gender_cut).all(axis=1)]

    data_table = data_table.astype(int).reset_index(drop=True)

    # Bring the table in the following format: col_0: anchor_1, col_1: anchor_2, col_2: odd-one-out
    for i, row in tqdm(
        iterable=data_table.iterrows(), desc=f"Prepare data for SPoSE & VICE in {session}", total=len(data_table)
    ):
        data_table.iloc[i, 0:3] = pd.value_counts(row, sort=True, ascending=True).index
    data_table = data_table.drop(columns=["head_odd"])
    data_table.columns = ["col_0", "col_1", "col_2"]

    # Replace head number with index
    index_mapper = (
        partial(head_nr_to_pilot_matrix_index, pilot_version=params.PILOT_VERSION)
        if pilot
        else head_nr_to_main_matrix_index
    )
    # pilot v2: female: 0-12, male: 13-25
    data_table = data_table.applymap(index_mapper)
    # for main: == data_table = data_table - 1

    if gender == "male":
        # When we have male-only triplets, we need to re-index the heads starting from 0 (instead of 50)
        data_table -= gender_cut

    sampled_index = None  # init
    if percentage is not None:
        data_table = data_table.sample(frac=percentage / 100)
        sampled_index = data_table.index
        data_table = data_table.reset_index(drop=True)

    # Extract training and test set (9-1-Ratio)
    training_set = data_table.sample(frac=0.9)
    test_set = data_table.drop(index=training_set.index)

    # Save training and test
    training_set.to_csv(p2_training_set, index=False, header=False, sep=" ")
    test_set.to_csv(p2_test_set, index=False, header=False, sep=" ")
    # Note: SPoSe takes .npy files as input, too
    np.save(file=p2_training_set.with_suffix(".npy"), arr=training_set.to_numpy())
    np.save(file=p2_test_set.with_suffix(".npy"), arr=test_set.to_numpy())

    if percentage is not None:
        # Save sampled index for a fraction of data
        np.save(file=spose_data_dir / "sampled_index.npy", arr=sampled_index)


def plot_weight_matrix(
    weights: np.ndarray, norm: bool, fig_name: str, save: bool = False, save_path: str | Path | None = ""
):
    """Plot the weight (i.e., `m`-dimensional embedding) matrix of `VICE` | `SPoSE`."""
    if norm:
        weights /= np.abs(weights).max()

    plt.matshow(weights, cmap="seismic", fignum=fig_name)  # could reduce to M_DIMENSIONS
    plt.colorbar()
    plt.tight_layout()

    if save:
        for ext in ["png", "svg"]:
            plt.savefig(Path(save_path, f"{fig_name}.{ext}"), dpi=300, format=ext)
        plt.close()
    else:
        plt.show()


def extract_faces_for_spose_dimensions(
    session: str,
    n_face: int | None = None,
    m_dims: int | None = None,
    pilot: bool = params.PILOT,
    return_path: bool = False,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, str]:
    """
    Extract the first `n` most representative faces for the first `m` dimensions of the trained SPoSE model.

    See Hebart et al. (2020), caption of Fig. 2:

    !!! quote
        'The images reflect the objects with the highest weights along those dimensions.'

    :param session: "2D", OR "3D"
    :param n_face: [int] restrict the number of faces OR [None] all faces are returned
    :param m_dims:  [int] restrict the number of dimensions (or weights) in the SPoSE model OR
                    [None] all dimensions are returned
    :param pilot: True: use pilot data
    :param return_path: True: return the path to the weights-file
    :return: indices of representative faces
    """
    # Load weights
    p2_weights = ""  # init
    weights = load_spose_weights(session=session, pilot=pilot, return_path=return_path, **kwargs)
    if return_path:
        weights, p2_weights = weights

    # Extract representative faces
    face_dim_idx_mat = np.argsort(weights, axis=0)[::-1][:n_face, :m_dims]
    # rows: index of most representative faces (descending) | cols: most relevant dimensions (descending)
    # E.g., face_img_idx[1, 0] = index of second most representative face for first dimension

    if return_path:
        return face_dim_idx_mat, p2_weights
    return face_dim_idx_mat


def extract_faces_for_vice_dimensions(
    session: str,
    n_face: int | None = None,
    m_dims: int | None = None,
    pilot: bool = params.PILOT,
    pruned: bool = True,
    return_path: bool = False,
    param_path: str | None = "",
) -> np.ndarray | tuple[np.ndarray, str]:
    """
    Extract the first `n` most representative faces for the first `m` dimensions of the trained VICE model.

    See Muttenthaler et al. (arXiv, 2022), p.19, Section F "Interpretability":

    !!! quote
        'Objects were sorted in descending order according to their absolute embedding value.'

    :param session: "2D", OR "3D"
    :param n_face: [int] restrict the number of faces OR [None] all faces are returned
    :param m_dims:  [int] restrict the number of dimensions (or weights) of the VICE model OR
                    [None] all dimensions are returned
    :param pilot: True: use pilot data
    :param pruned: True: return the pruned parameters
    :param return_path: True: return path to the parameter file
    :param param_path: path to weight file, defined by the corresponding VICE params (after /[session]/..)
    :return: indices of representative faces
    """
    # Load weights
    p2_weights = ""  # init
    weights = load_vice_weights(
        session=session, pilot=pilot, pruned=pruned, return_path=return_path, param_path=param_path
    )
    if return_path:
        loc_param, _, p2_weights = weights  # scale_param = _
    else:
        loc_param, _ = weights  # _ = scale_param

    # # Extract representative faces
    # In the paper they report taking the 'absolute embedding value' to sort objects.
    # However, in facesim3d.modeling.VICE.visualization.plot_topk_objects_per_dimension(), objects are
    # just sorted based on the corresponding weight-value (mu, not sigma), so we do this here, too.
    # Bt also check out: np.linalg.norm(loc_param, axis=0), dimensions are here semi-sorted.
    n_face = loc_param.shape[0] if n_face is None else n_face
    m_dims = loc_param.shape[1] if m_dims is None else m_dims
    face_dim_idx_mat = np.argsort(loc_param, axis=0)[::-1][:n_face, :m_dims]
    # face_dim_idx_mat = np.argsort(np.abs(loc_param), axis=0)[::-1][:n_face, :m_dims]  # noqa: ERA001
    # Note, that taking the absolute didn't change the results anyway (tested for '2D')
    # rows: index of most representative faces (descending) | cols: most relevant dimensions (descending)
    # E.g., face_img_idx[1, 0] = index of second most representative face for first dimension

    if return_path:
        return face_dim_idx_mat, p2_weights
    return face_dim_idx_mat


def display_representative_faces(
    face_dim_idx_mat: np.ndarray,
    pilot: bool = params.PILOT,
    as_grid: bool = True,
    title: str | None = None,
    dim_indices: list | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Display representative faces for the first `m` dimensions of the trained sparse model.

    :param face_dim_idx_mat: nxm matrix of indices in representative faces per dimension
    :param pilot: True: use pilot data
    :param as_grid: plot as image grid
    :param title: Image or figure title
    :param dim_indices: list of dimension indices to display
    :param save_path: if the path is given, save figure
    :return: None
    """
    # Choose the map for CFD indices
    idx_mapper = (
        partial(pilot_matrix_index_to_head_nr, pilot_version=params.PILOT_VERSION)
        if pilot
        else main_matrix_index_to_head_nr
    )

    fig, axes = None, None  # init
    if as_grid:
        dim_indices = range(face_dim_idx_mat.shape[1]) if dim_indices is None else dim_indices
        fig, axes = plt.subplots(
            *face_dim_idx_mat.shape,
            figsize=(face_dim_idx_mat.shape[1] * 2, face_dim_idx_mat.shape[0] * 2),
            sharex=True,
            sharey=True,
            num=title,
        )

    for dim_i, face_indices in enumerate(face_dim_idx_mat.T):
        print(dim_i, face_indices)

        for i, face_idx in enumerate(face_indices):
            # Map indices back to CFD index
            face_id = idx_mapper(face_idx)

            # Display faces with the strongest weight(s) for each dimension
            if as_grid:
                try:
                    face_img = plt.imread(fname=face_image_path(head_id=face_id))  # load as np.array
                except FileNotFoundError:
                    # This computes the screenshot of the 3D reconstructed face
                    display_face(head_id=face_id, data_mode="3d-reconstructions", interactive=False, verbose=False)
                    face_img = plt.imread(fname=face_image_path(head_id=face_id))

                # Zooming takes a while; now it is done already at earlier stage in display_face()
                # face_img = clipped_zoom(img=face_img, zoom_factor=1.8)  # zoom into image  # noqa: ERA001
                axes[i, dim_i].imshow(face_img)
                if i == 0:
                    axes[i, dim_i].set_title(f"Dimension {dim_indices[dim_i] + 1}")
                axes[i, dim_i].set_xticks([])
                axes[i, dim_i].set_xlabel(face_id)
                axes[i, dim_i].yaxis.set_visible(False)
                for spine in axes[i, dim_i].spines.values():  # remove axes-box around image
                    spine.set_visible(False)
            else:
                # Display face images externally
                # TODO: add somewhere a caption  # noqa: FIX002
                display_face(head_id=face_id)
    if as_grid:
        fig.tight_layout()
        if save_path:
            fn = (
                f"{title}_representative_faces-{face_dim_idx_mat.shape[0]}_"
                f"dims-{str(tuple(np.array(dim_indices) + 1)).replace(' ', '')}"
            )
            cprint(string=f"Saving figure in '{Path(str(save_path), fn)}' ... ", col="b")
            for ext in ["png", "svg"]:
                plt.savefig(fname=Path(save_path, fn).with_suffix("." + ext), dpi=300, format=ext)
            plt.close()


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # 0) Set up logging
    logger = logging.getLogger(__name__)  # get predefined logger
    logger_filename = Path(
        paths.results.pilot.V2 if params.PILOT else paths.results.MAIN, "logs", "computational_choice_model.log"
    )
    if OVERWRITE_LOGGER and logger_filename.is_file():
        logger_filename.unlink()  # delete old log file

    logger = update_logger_configs(
        new_logger_name="ComputationalChoiceModels",
        new_logger_filename=logger_filename,
        logger=logger,
    )

    # 1) Prepare data for SPoSE & VICE
    prepare_data_for_spose_and_vice(session="2D", pilot=params.PILOT)
    prepare_data_for_spose_and_vice(session="3D", pilot=params.PILOT)

    # Prepare gender specific data
    prepare_data_for_spose_and_vice(session="2D", gender=True, pilot=params.PILOT)
    prepare_data_for_spose_and_vice(session="3D", gender=True, pilot=params.PILOT)

    #  >><< o >><< o >>><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # 2) Run SPoSE & VICE model via bash script
    # cd FaceSim3D/
    # bash .../[SPoSE|VICE]/bash_for_training/training_[spose|vice]_pilot_v2.sh  # in case of pilot or ...
    # bash .../[SPoSE|VICE]/bash_for_training/training_[spose|vice]_main_hp_search.sh  # HP search

    # Choose the best hyperparameters
    # SPoSE
    n_to_print_per_session: int = 1
    for sess in params.SESSIONS:
        cprint(string=f"\nBest SPoSE lambda (hyperparameter; descending) | Session: {sess}", col="b", fm="ul")
        spose_model_table = list_spose_model_performances(session=sess, gender=None)
        for ctn, p2_model in enumerate(spose_model_table.model_path.values):
            print()
            # Extract lambda from the model path
            print("\tLambda:", p2_model.split("/")[-2])
            if ctn + 1 == n_to_print_per_session:
                break

    # VICE
    BEST_HP_VICE = get_best_hp_vice(hp_search=True, print_n=n_to_print_per_session)
    # take `hp_search=True`, since the best hyperparameters are determined in the HP-search.
    if not HP_SEARCH:
        for sess in params.SESSIONS:
            BEST_HP_VICE[sess].pop("hp_perc")

    # Run on full dataset with the best hyperparameters
    # bash .../[SPoSE|VICE]/bash_for_training/training_[spose|vice]_main.sh # in case of the main study

    #  >><< o >><< o >>><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # 3) Extract representative faces
    # for SPoSE model weights
    mat_spose_2d, p2_spose_weights_2d = extract_faces_for_spose_dimensions(
        session="2D",
        n_face=None,
        m_dims=M_DIMENSIONS,
        pilot=params.PILOT,
        return_path=True,
        hp=BEST_HP_SPOSE["2D"],
        gender=None,
    )
    mat_spose_3d, p2_spose_weights_3d = extract_faces_for_spose_dimensions(
        session="3D",
        n_face=None,
        m_dims=M_DIMENSIONS,
        pilot=params.PILOT,
        return_path=True,
        hp=BEST_HP_SPOSE["3D"],
        gender=None,
    )

    # Print performance on last epoch
    mea_table = get_mea_table()
    for sess, spose_path in zip(params.SESSIONS, [p2_spose_weights_2d, p2_spose_weights_3d], strict=True):
        result_dir = Path(spose_path).parent
        result_fn = max(f for f in os.listdir(result_dir) if f.startswith("results"))
        with (result_dir / result_fn).open() as f:
            result = f.read()

        val_acc = json.loads(result)["val_acc"]
        epoch = json.loads(result)["epoch"]
        mea_sess = mea_table.loc[(sess, "multi-sub-sample")].max_acc

        # Normalize the model performance to the maximal empirical accuracy (noise ceiling) and chance level
        normed_val_performance = normalize(
            val_acc, lower_bound=0.0, upper_bound=1.0, global_min=CHANCE_LVL, global_max=mea_sess
        )
        msg = (
            f"Validation accuracy (on epoch {epoch}) of SPoSE model in {sess} session: "
            f"{val_acc:.2%} (MEA-chance-level-normed: {normed_val_performance:.4}, where MEA={mea_sess:.2%})"
        )
        print(msg)
        logger.info(msg)

    # for VICE model parameters
    # Get paths to VICE parameters
    param_path_vice_2d = create_path_from_vice_params(params_dict=BEST_HP_VICE["2D"], gender=None, pilot=params.PILOT)
    param_path_vice_2d = str(param_path_vice_2d).split("VICE/2D/")[-1]
    # == PATH_TO_VICE_WEIGHTS.format(**BEST_HP_VICE["2D"])
    param_path_vice_3d = create_path_from_vice_params(params_dict=BEST_HP_VICE["3D"], gender=None, pilot=params.PILOT)
    param_path_vice_3d = str(param_path_vice_3d).split("VICE/3D/")[-1]

    mat_vice_2d, p2_vice_weights_2d = extract_faces_for_vice_dimensions(
        n_face=None,
        m_dims=M_DIMENSIONS,
        session="2D",
        pilot=params.PILOT,
        pruned=True,
        return_path=True,
        param_path=param_path_vice_2d,
    )

    mat_vice_3d, p2_vice_weights_3d = extract_faces_for_vice_dimensions(
        n_face=None,
        m_dims=M_DIMENSIONS,
        session="3D",
        pilot=params.PILOT,
        pruned=True,
        return_path=True,
        param_path=param_path_vice_3d,
    )

    # Print performance on last epoch
    for sess, vice_path in zip(params.SESSIONS, [p2_vice_weights_2d, p2_vice_weights_3d], strict=True):
        result_dir = Path(vice_path).parent
        result_fn = max(f for f in os.listdir(result_dir) if f.startswith("results"))
        with (result_dir / result_fn).open() as f:
            result = f.read()

        val_acc = json.loads(result)["val_acc"]
        epoch = json.loads(result)["epoch"]
        mea_sess = mea_table.loc[(sess, "multi-sub-sample")].max_acc

        # Normalize the model performance to the maximal empirical accuracy (noise ceiling) and chance level
        normed_val_performance = normalize(
            val_acc, lower_bound=0.0, upper_bound=1.0, global_min=CHANCE_LVL, global_max=mea_sess
        )

        msg = (
            f"Validation accuracy (on epoch {epoch}) of VICE model in {sess} session: "
            f"{val_acc:.2%} (MEA-chance-level-normed: {normed_val_performance:.4}, where MEA={mea_sess:.2%})"
        )
        logger.info(msg)

    #  >><< o >><< o >>><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # 4) Compute similarity matrix of VICE weights (this is partially done in rsa.py, including SPoSE)
    # Compute similarity matrix of VICE weights (only mu at index [0])
    vice_weights_2d = load_vice_weights(
        session="2D",
        pilot=params.PILOT,
        pruned=True,
        return_path=False,
        param_path=param_path_vice_2d,
    )[0]  # take only loc_params

    vice_sim_mat_2d = compute_vice_similarity_matrix(
        session="2D",
        gender=None,
        hp_search=HP_SEARCH,
        pilot=params.PILOT,
        save=False,
        verbose=True,  # saved in rsa.py
    )

    vice_weights_3d = load_vice_weights(
        session="3D",
        pilot=params.PILOT,
        pruned=True,
        return_path=False,
        param_path=param_path_vice_3d,
    )[0]  # take only loc_params

    vice_sim_mat_3d = compute_vice_similarity_matrix(
        session="3D", gender=None, hp_search=HP_SEARCH, pilot=params.PILOT, save=False, verbose=True
    )

    # Explore similarities based on VICE in individual faces (for pilot)
    # display_face(main_matrix_index_to_head_nr(face_idx=95)) # VICE: man # noqa: ERA001
    # display_face(main_matrix_index_to_head_nr(face_idx=23)) # ... similar to woman idx=23 # noqa: ERA001

    corr_func = spearmanr if SPEARMAN else pearsonr
    r, p = corr_func(vectorize_similarity_matrix(vice_sim_mat_2d), vectorize_similarity_matrix(vice_sim_mat_3d))
    msg = f"RSA similarity matrices of VICE 2D-3D: R={r:.3f}, p<={p:.3f} - \tUnexplained variance: {1 - r**2:.2%}"
    logger.info(msg)

    #  >><< o >><< o >>><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # 5) Load PFA table and correlate with weight matrix
    idx_to_model_mapper = (
        partial(pilot_index_to_model_name, pilot_version=params.PILOT_VERSION)
        if params.PILOT
        else main_index_to_model_name
    )
    idx_mapper = (
        partial(pilot_matrix_index_to_head_nr, pilot_version=params.PILOT_VERSION)
        if params.PILOT
        else main_matrix_index_to_head_nr
    )
    pfa_tab = get_cfd_features_for_models(list_of_models=list(map(idx_to_model_mapper, range(n_faces))))
    pfa_mat = pfa_tab.to_numpy().astype(float)
    pfa_labels = list(map(cfd_var_converter, pfa_tab.columns.to_numpy()))

    # 2D
    vice_pfa_corr_mat_2d = compute_pearson_correlation_between_two_feature_matrices(x=vice_weights_2d, y=pfa_mat)

    visualise_matrix(
        vice_pfa_corr_mat_2d,
        session="2D",
        pilot=params.PILOT,
        use_rsatoolbox=vice_pfa_corr_mat_2d.shape[0] == vice_pfa_corr_mat_2d.shape[1],
        fig_name="Correlation of VICE weights and PFA matrix in 2D",
        cmap="seismic",
        save=True,
        save_path=Path(p2_vice_weights_2d).parent,
        xticklabels=pfa_labels,
        # xlabel="Physical face attributes",  # noqa: ERA001
        ylabel="VICE 2D dimensions",
        title="Correlation of VICE weights and Physical Face Attributes (PFA) in 2D",
    )

    msg = f"Positive correlation (Pearson's R >= {CORR_TH})"
    logger.info(msg)
    for w_i, p_j in zip(*np.where(vice_pfa_corr_mat_2d >= CORR_TH), strict=True):
        msg = (
            f"VICE (2D) dim-{w_i + 1} correlates (R={vice_pfa_corr_mat_2d[w_i, p_j]:.2f}) with "
            f"'{cfd_var_converter(pfa_tab.columns[p_j])}' at index {p_j}"
        )
        logger.info(msg)
        dim_max_face_idx = int(np.argmax(vice_weights_2d[:, w_i]))
        dim_min_face_idx = int(np.argmin(vice_weights_2d[:, w_i]))
        msg = (
            f"\tVICE (2D) max weight of Dim-{w_i + 1} has '{idx_mapper(dim_max_face_idx)}' "
            f"cp. to: '{cfd_var_converter(pfa_tab.columns[p_j])}' = "
            f"{pfa_tab.loc[idx_to_model_mapper(dim_max_face_idx), pfa_tab.columns[p_j]]}"
        )
        logger.info(msg)
        msg = (
            f"\tVICE (2D) min weight of Dim-{w_i + 1} has '{idx_mapper(dim_min_face_idx)}' "
            f"cp. to: '{cfd_var_converter(pfa_tab.columns[p_j])}' = "
            f"{pfa_tab.loc[idx_to_model_mapper(dim_min_face_idx), pfa_tab.columns[p_j]]}"
        )
        logger.info(msg)

    msg = f"Negative correlation (Pearson's R <= -{CORR_TH})"
    logger.info(msg)
    for w_i, p_j in zip(*np.where(vice_pfa_corr_mat_2d <= -CORR_TH), strict=True):
        msg = (
            f"VICE (2D) dim-{w_i + 1} neg. correlates (R={vice_pfa_corr_mat_2d[w_i, p_j]:.2f}) with "
            f"'{cfd_var_converter(pfa_tab.columns[p_j])}' at index {p_j}"
        )
        logger.info(msg)

        dim_max_face_idx = int(np.argmax(vice_weights_2d[:, w_i]))
        dim_min_face_idx = int(np.argmin(vice_weights_2d[:, w_i]))
        msg = (
            f"\tVICE (2D) max weight of Dim-{w_i + 1} has '{idx_mapper(dim_max_face_idx)}' "
            f"cp. to: '{cfd_var_converter(pfa_tab.columns[p_j])}' = "
            f"{pfa_tab.loc[idx_to_model_mapper(dim_max_face_idx), pfa_tab.columns[p_j]]}"
        )
        logger.info(msg)
        msg = (
            f"\tVICE (2D) min weight of Dim-{w_i + 1} has '{idx_mapper(dim_min_face_idx)}' "
            f"cp. to: '{cfd_var_converter(pfa_tab.columns[p_j])}' = "
            f"{pfa_tab.loc[idx_to_model_mapper(dim_min_face_idx), pfa_tab.columns[p_j]]}"
        )
        logger.info(msg)

    # Correlate with gender
    msg = "Correlation with gender"
    logger.info(msg)
    n_female = 12 if (params.PILOT and params.PILOT_VERSION == 2) else n_faces // 2  # noqa: PLR2004
    v_gender = np.array([0] * n_female + [1] * (n_faces - n_female))  # create gender vector (0-1-coding)

    # First, PFA ~ gender
    for p_j in range(pfa_mat.shape[1]):
        r, p = pearsonr(pfa_mat[:, p_j], v_gender)
        left_string = f"PFA '{cfd_var_converter(pfa_tab.columns[p_j])}' ~ gender:"
        right_string = f"R={r:.3f}, p<={p:.3f}"
        msg = f"{left_string:<35} {right_string:>18}"
        logger.info(msg)

    # VICE 2D ~ gender
    for w_i in range(vice_weights_2d.shape[1]):
        r, p = pearsonr(vice_weights_2d[:, w_i], v_gender)
        right_string = f"R={r:.3f}, p<={p:.3f}"
        msg = f"VICE-2D dim {w_i + 1:2d} ~ gender:\t{right_string:>18}"
        logger.info(msg)

    # Plot VICE weight matrix (2D) for comparison
    plot_weight_matrix(
        weights=vice_weights_2d,
        norm=True,
        fig_name="2D - VICE weights",
        save=True,
        save_path=Path(p2_vice_weights_2d).parent,
    )

    # 3D
    vice_pfa_corr_mat_3d = compute_pearson_correlation_between_two_feature_matrices(x=vice_weights_3d, y=pfa_mat)
    visualise_matrix(
        vice_pfa_corr_mat_3d,
        session="3D",
        pilot=params.PILOT,
        use_rsatoolbox=vice_pfa_corr_mat_3d.shape[0] == vice_pfa_corr_mat_3d.shape[1],
        fig_name="Correlation of VICE weights and PFA matrix in 3D",
        cmap="seismic",
        save=True,
        save_path=Path(p2_vice_weights_3d).parent,
        xticklabels=pfa_labels,
        # xlabel="Physical face attributes",  # noqa: ERA001
        ylabel="VICE 3D dimensions",
        title="Correlation of VICE weights and Physical Face Attributes (PFA) in 3D",
    )

    msg = f"Positive correlation (Pearson's R >= {CORR_TH})"
    logger.info(msg)
    for w_i, p_j in zip(*np.where(vice_pfa_corr_mat_3d >= CORR_TH), strict=True):
        msg = (
            f"VICE (3D) Dim-{w_i + 1} correlates (R={vice_pfa_corr_mat_3d[w_i, p_j]:.2f}) with "
            f"'{cfd_var_converter(pfa_tab.columns[p_j])}' (at index {p_j})"
        )
        logger.info(msg)
        dim_max_face_idx = int(np.argmax(vice_weights_3d[:, w_i]))
        dim_min_face_idx = int(np.argmin(vice_weights_3d[:, w_i]))
        msg = (
            f"\tVICE (3D) max weight of Dim-{w_i + 1} has '{idx_mapper(dim_max_face_idx)}' "
            f"cp. to: '{cfd_var_converter(pfa_tab.columns[p_j])}' = "
            f"{pfa_tab.loc[idx_to_model_mapper(dim_max_face_idx), pfa_tab.columns[p_j]]}"
        )
        logger.info(msg)
        msg = (
            f"\tVICE (3D) min weight of Dim-{w_i + 1} has '{idx_mapper(dim_min_face_idx)}' "
            f"cp. to: '{cfd_var_converter(pfa_tab.columns[p_j])}' = "
            f"{pfa_tab.loc[idx_to_model_mapper(dim_min_face_idx), pfa_tab.columns[p_j]]}"
        )
        logger.info(msg)

    msg = f"Negative correlation (Pearson's R <= -{CORR_TH})"
    logger.info(msg)
    for w_i, p_j in zip(*np.where(vice_pfa_corr_mat_3d <= -CORR_TH), strict=True):
        msg = (
            f"VICE (3D) Dim-{w_i + 1} neg. correlates (R={vice_pfa_corr_mat_3d[w_i, p_j]:.2f}) with "
            f"'{cfd_var_converter(pfa_tab.columns[p_j])}' (at index {p_j})"
        )
        logger.info(msg)
        dim_max_face_idx = int(np.argmax(vice_weights_3d[:, w_i]))
        dim_min_face_idx = int(np.argmin(vice_weights_3d[:, w_i]))
        msg = (
            f"\tVICE (3D) max weight of Dim-{w_i + 1} has '{idx_mapper(dim_max_face_idx)}' "
            f"cp. to: '{cfd_var_converter(pfa_tab.columns[p_j])}' = "
            f"{pfa_tab.loc[idx_to_model_mapper(dim_max_face_idx), pfa_tab.columns[p_j]]}"
        )
        logger.info(msg)
        msg = (
            f"\tVICE (3D) min weight of Dim-{w_i + 1} has '{idx_mapper(dim_min_face_idx)}' "
            f"cp. to: '{cfd_var_converter(pfa_tab.columns[p_j])}' = "
            f"{pfa_tab.loc[idx_to_model_mapper(dim_min_face_idx), pfa_tab.columns[p_j]]}"
        )
        logger.info(msg)

    # Correlate with gender
    for w_i in range(vice_weights_3d.shape[1]):
        r, p = pearsonr(vice_weights_3d[:, w_i], v_gender)
        right_string = f"R={r:.3f}, p<={p:.3f}"
        msg = f"VICE-3D dim {w_i + 1:2d} ~ gender:\t{right_string:>18}"
        logger.info(msg)

    # Plot VICE weight matrix (3D) for comparison
    plot_weight_matrix(
        weights=vice_weights_3d,
        norm=True,
        fig_name="3D - VICE weights",
        save=True,
        save_path=Path(p2_vice_weights_3d).parent,
    )

    # Explore similarities based on VICE in individual faces
    # display_face("Head3", data_mode="3d-reconstructions")  # putative big eyes, idx_mapper(2)  # noqa: ERA001
    # display_face("Head55", data_mode="3d-reconstructions")  # ... small eyes idx_mapper(16)  # noqa: ERA001
    # display_face("Head57", data_mode="2d-original")  # putative wide nose  (see 2D)  # noqa: ERA001
    # display_face("Head5", data_mode="2d-original")  # ... thin nose  # noqa: ERA001
    # display_face(idx_mapper(1), data_mode="3d-reconstructions")  # female most different to others  # noqa: ERA001

    #  >><< o >><< o >>><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # 6) Display representative faces for SPoSE & VICE model weights
    # Could add the least representative face(s) per dimension

    # Display faces for SPoSE
    # Manually select dimensions for SPoSE
    mat_spose_selected_2d = mat_spose_2d[0:N_REPR_FACES, SPOSE_DIM_IND_2D or slice(None)]
    mat_spose_selected_3d = mat_spose_3d[0:N_REPR_FACES, SPOSE_DIM_IND_3D or slice(None)]

    # Most representative faces
    display_representative_faces(
        face_dim_idx_mat=mat_spose_selected_2d,
        pilot=params.PILOT,
        as_grid=True,
        dim_indices=SPOSE_DIM_IND_2D or None,
        title="2D_most",
        save_path=Path(p2_spose_weights_2d).parent,
    )
    display_representative_faces(
        face_dim_idx_mat=mat_spose_selected_3d,
        pilot=params.PILOT,
        as_grid=True,
        dim_indices=SPOSE_DIM_IND_3D or None,
        title="3D_most",
        save_path=Path(p2_spose_weights_3d).parent,
    )

    # Least representative faces
    mat_spose_selected_2d = mat_spose_2d[-N_REPR_FACES:, SPOSE_DIM_IND_2D or slice(None)]
    mat_spose_selected_3d = mat_spose_3d[-N_REPR_FACES:, SPOSE_DIM_IND_3D or slice(None)]

    display_representative_faces(
        face_dim_idx_mat=mat_spose_selected_2d,
        pilot=params.PILOT,
        as_grid=True,
        dim_indices=SPOSE_DIM_IND_2D or None,
        title="2D_least",
        save_path=Path(p2_spose_weights_2d).parent,
    )
    display_representative_faces(
        face_dim_idx_mat=mat_spose_selected_3d,
        pilot=params.PILOT,
        as_grid=True,
        dim_indices=SPOSE_DIM_IND_3D or None,
        title="3D_least",
        save_path=Path(p2_spose_weights_3d).parent,
    )

    # Display faces for VICE
    # Check out: from facesim3d.modeling.VICE.visualization import plot_topk_objects_per_dimension
    mat_vice_selected_2d = mat_vice_2d[0:N_REPR_FACES, VICE_DIM_IND_2D or slice(None)]
    mat_vice_selected_3d = mat_vice_3d[0:N_REPR_FACES, VICE_DIM_IND_3D or slice(None)]

    # Most representative faces
    display_representative_faces(
        face_dim_idx_mat=mat_vice_selected_2d,
        pilot=params.PILOT,
        as_grid=True,
        dim_indices=VICE_DIM_IND_2D or None,
        title="2D_most",
        save_path=Path(p2_vice_weights_2d).parent,
    )
    display_representative_faces(
        face_dim_idx_mat=mat_vice_selected_3d,
        pilot=params.PILOT,
        as_grid=True,
        dim_indices=VICE_DIM_IND_3D or None,
        title="3D_most",
        save_path=Path(p2_vice_weights_3d).parent,
    )

    # Least representative faces
    mat_vice_selected_2d = mat_vice_2d[-N_REPR_FACES:, VICE_DIM_IND_2D or slice(None)]
    mat_vice_selected_3d = mat_vice_3d[-N_REPR_FACES:, VICE_DIM_IND_3D or slice(None)]

    display_representative_faces(
        face_dim_idx_mat=mat_vice_selected_2d,
        pilot=params.PILOT,
        as_grid=True,
        dim_indices=VICE_DIM_IND_2D or None,
        title="2D_least",
        save_path=Path(p2_vice_weights_2d).parent,
    )
    display_representative_faces(
        face_dim_idx_mat=mat_vice_selected_3d,
        pilot=params.PILOT,
        as_grid=True,
        dim_indices=VICE_DIM_IND_3D or None,
        title="3D_least",
        save_path=Path(p2_vice_weights_3d).parent,
    )

    # Plot dimensions with the n most representative faces (full matrices)
    # plt.matshow(mat_spose_2d, cmap="BrBG", fignum="2D - SPoSE")  # noqa: ERA001
    # plt.matshow(mat_spose_3d, cmap="BrBG", fignum="3D - SPoSE")  # noqa: ERA001
    # plt.matshow(mat_vice_2d, cmap="BrBG", fignum="2D - VICE")  # noqa: ERA001
    # plt.matshow(mat_vice_3d, cmap="BrBG", fignum="3D - VICE")  # noqa: ERA001

    # Same as print out
    # cprint(f"2D-SPoSE\n{mat_spose_2d}", col='y')  # noqa: ERA001
    # cprint(f"\n3D-SPoSE\n{mat_spose_3d}\n", col='y')  # noqa: ERA001
    # cprint(f"2D-VICE\n{mat_vice_2d}", col='y')  # noqa: ERA001
    # cprint(f"\n3D-VICE\n{mat_vice_3d}", col='y')  # noqa: ERA001

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
