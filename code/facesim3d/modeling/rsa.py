# !/usr/bin/env python3
"""
Compute RDMs of face similarity judgments and run RSA for the pilot study (version 2) & the main experiment.

Run the script with (see also `./results/main/rsa/run_rsa.sh`):

``` bash
for metric in cosine euclidean
do
    python -m facesim3d.modeling.rsa --metric ${metric} --save_corr --plot --save_plots --logger_overwrite -v
done
```

??? question "Need information about input arguments?"
    See the help section:
    ``` bash
    python -m facesim3d.modeling.rsa --help
    ```
"""

# %% Import
from __future__ import annotations

import argparse
import itertools
import logging
from functools import lru_cache, partial
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rsatoolbox
import seaborn as sns
from matplotlib.ticker import FixedLocator
from numpy import typing as npt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from ut.ils import cprint, send_to_mattermost

from facesim3d.configs import config, params, paths, update_logger_configs
from facesim3d.modeling.compute_similarity import (
    compute_feature_similarity_matrix,
)
from facesim3d.modeling.face_attribute_processing import (
    get_cfd_features_for_models,
    head_nr_to_pilot_matrix_index,
    heads_naming_converter_table,
)
from facesim3d.modeling.FLAME.extract_flame_params import get_flame_params, latent_flame_code
from facesim3d.modeling.prep_computational_choice_model import (
    BEST_HP_SPOSE,
    create_path_from_vice_params,
    get_best_hp_vice,
    load_spose_weights,
    load_vice_weights,
)
from facesim3d.modeling.VGG.extract_vgg_feature_maps import (
    get_vgg_activation_maps,
    get_vgg_human_judgment_activation_maps,
)
from facesim3d.modeling.VGG.models import (
    get_vgg_face_model,
    get_vgg_performance_table,
    load_trained_vgg_face_human_judgment_model,
)
from facesim3d.modeling.VGG.prepare_data import prepare_data_for_human_judgment_model
from facesim3d.read_data import read_pilot_data, read_trial_results_of_participant, read_trial_results_of_session

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# Vars
PCA_FRACTION: float = 0.9  # fraction of variance to be explained by PCA
SPEARMAN: bool = True  # use spearman correlation instead of pearson
# Note Jozwik et al. (2022) use Pearson correlation,
# However, according to https://academic.oup.com/scan/article/14/11/1243/5693905
# Spearman is recommended for comparing RDMs. Also, Kietzmann et al. use Spearman (see talk @ CCN 2023).
FLAME_MODEL: str = "deca"
METRIC: str = "cosine"  # "euclidean" OR "cosine"

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def check_gender(gender: str) -> str:
    """Check whether the `gender` argument is valid."""
    gender = gender.lower()
    if gender not in params.GENDERS:
        msg = f"gender must be in {params.GENDERS}"
        raise ValueError(msg)
    return gender


@lru_cache(maxsize=6)
def get_model_hps(
    session: str,
    model_name: str | None = None,
    data_mode: str | None = None,
    exclusive_gender_trials: str | None = None,
) -> pd.Series:
    """
    Get model hyperparameters.

    :param session: 2D, OR 3D
    :param model_name: name of the model
    :param data_mode: '2d-original', '3d-reconstructions', OR '3d-perspectives'
    :param exclusive_gender_trials: 'female', 'male', OR None for all trials.
    :return: model-specific hyperparameters
    """
    # Get the model performance table
    model_table = get_vgg_performance_table(
        sort_by_acc=True, hp_search=False, exclusive_gender_trials=exclusive_gender_trials
    )

    # Filter for session
    model_table = model_table[model_table.session == session]

    # Filter for data mode
    if data_mode is not None:
        data_mode = data_mode.lower()
        model_table = model_table[model_table.data_mode == data_mode]

    # Filter for the model
    if model_name is None:
        # Take the best model
        model_hp_row = model_table[model_table.model_name == model_table.model_name].iloc[0]
    else:
        # Take the requested model
        model_hp_row = model_table[model_table.model_name == model_name].iloc[0]

    return model_hp_row


def compute_physical_attr_similarity_matrix(
    pca: bool | float = False,
    gender: str | None = None,
    pilot_version: int | None = None,
    metric: str = "cosine",
) -> npt.NDArray[np.float64]:
    """
    Compute a similarity matrix from physical face attributes (`PFA`).

    !!! note "Expected performance of the `PFA` model"
        Jozwik et al. (2022) report "poor performance of configural models", which are similar to `PFA` here,
        however, good performance of the active appearance model (`AAM`) which is based on similar features.

    :param pca: False OR provide (0.< pca < 1.) if PCA should be run on feature table with n components such that
                pca [float] *100 % of variance is explained
    :param gender: for within-gender feature comparison
    :param pilot_version: None for the main experiment, OR pilot 1, OR 2.
    :param metric: similarity metric to use (cosine, Euclidean)
    :return: similarity matrix of physical face attributes

    """
    # Get head mapping table
    head_map = heads_naming_converter_table(pilot_version=pilot_version)
    list_of_models = head_map.Model

    # Filter for gender if requested
    if gender is not None:
        gender = check_gender(gender)
        list_of_models = head_map.Model[head_map.Model.str.contains("WF" if "female" in gender else "WM")]

    # Get table with physical attributes/features
    feat_tab = get_cfd_features_for_models(list_of_models=list_of_models, physical_attr_only=True)

    return compute_feature_similarity_matrix(feature_table=feat_tab, pca=pca, metric=metric, z_score=True)


def compute_flame_feature_similarity_matrix(
    pca: bool | float = False,
    param: str = "shape",
    model: str = "deca",
    gender: str | None = None,
    pilot_version: int | None = None,
    metric: str = "cosine",
) -> npt.NDArray[np.float64]:
    """
    Compute a similarity matrix from (e.g., shape) parameters of the `FLAME`-fitted heads.

    :param pca: False OR provide (0.< pca < 1.) if PCA should be run on feature table with n components such that
                pca [float] *100 % of variance is explained
    :param param: which parameter to use (e.g., 'shape', 'exp' [expression], 'pose', ...)
    :param model: which model to use: 'deca' OR 'flame'
    :param gender: for exclusively within-gender feature comparison
    :param pilot_version: None for main experiment,OR pilot 1, OR 2.
    :param metric: similarity metric to use (cosine, Euclidean)
    :return: similarity matrix based on shape parameters of the FLAME-fitted heads
    """
    # Get head mapping table
    head_map = heads_naming_converter_table(pilot_version=pilot_version)

    # Filter for gender if requested
    if gender is not None:
        gender = check_gender(gender=gender)
        head_map = head_map[head_map.Model.str.contains("WF" if gender == "female" else "WM")]

    # Get table with FLAME shape parameters for all heads
    feat_tab = get_flame_params(list_of_head_nrs=head_map.head_nr, param=param, model=model)

    return compute_feature_similarity_matrix(feature_table=feat_tab, pca=pca, metric=metric, z_score=True)


def compute_spose_feature_map_similarity_matrix(
    session: str,
    pca: bool | float = False,
    gender: str | None = None,
    pilot_version: int | None = None,
    metric: str = "cosine",
) -> npt.NDArray[np.float64]:
    """
    Compute a similarity matrix from `SPoSE` feature maps (embedding matrix).

    :param session: '2D', OR '3D'
    :param pca: False OR provide (0.< pca < 1.) if PCA should be run on feature table with n components such that
                pca [float] *100 % of variance is explained
    :param gender: for exclusively within-gender feature comparison
    :param pilot_version: None for the main experiment, OR pilot 1, OR 2.
    :param metric: similarity metric to use (cosine, Euclidean)
    :return: similarity matrix of SPoSE feature maps
    """
    if pilot_version == 1:
        msg = "SPoSE feature maps are not available for pilot 1."
        raise ValueError(msg)

    # Get head mapping table
    head_map = heads_naming_converter_table(pilot_version=pilot_version)
    list_of_models = head_map.Model

    # Filter for gender if requested
    if gender is not None:
        # Prepare gender-only features if requested
        gender = check_gender(gender)
        list_of_models = head_map.Model[head_map.Model.str.contains("WF" if "female" in gender else "WM")]

    # Get the weights of the best hyperparameters for SPoSE
    spose_weights = load_spose_weights(
        session=session,
        gender=gender,
        pilot=pilot_version is not None,
        return_path=False,
        **BEST_HP_SPOSE[session],
    )

    feat_tab = pd.DataFrame(index=list_of_models, columns=[f"D{i:03d}" for i in range(spose_weights.shape[1])])
    feat_tab.loc[:, :] = spose_weights

    # with z_score=False: == compute_spose_similarity_matrix
    return compute_feature_similarity_matrix(feature_table=feat_tab, pca=pca, metric=metric, z_score=False)


def compute_vice_feature_map_similarity_matrix(
    session: str,
    pca: bool | float = False,
    gender: str | None = None,
    pilot_version: int | None = None,
    metric: str = "cosine",
) -> npt.NDArray[np.float64]:
    """
    Compute a similarity matrix from `VICE` feature maps (embedding matrix).

    :param session: '2D', OR '3D'
    :param pca: False OR provide (0.< pca < 1.) if PCA should be run on feature table with n components such that
                pca [float] *100 % of variance is explained
    :param gender: for exclusively within-gender feature comparison
    :param pilot_version: None for the main experiment, OR pilot 1, OR 2.
    :param metric: similarity metric to use (cosine, Euclidean)
    :return: similarity matrix of VICE feature maps
    """
    if pilot_version == 1:
        msg = "VICE feature maps are not available for pilot 1."
        raise ValueError(msg)

    # Get head mapping table
    head_map = heads_naming_converter_table(pilot_version=pilot_version)
    list_of_models = head_map.Model

    # Filter for gender if requested
    if gender is not None:
        # Prepare gender-only features if requested
        gender = check_gender(gender)
        list_of_models = head_map.Model[head_map.Model.str.contains("WF" if "female" in gender else "WM")]

    # Get the weights of the best hyperparameters for VICE
    best_hp_vice = get_best_hp_vice(hp_search=True, print_n=0, from_config=True)[session]
    best_hp_vice.pop("hp_perc")  # we want params of the main run
    path_to_vice_sim_mat = create_path_from_vice_params(
        params_dict=best_hp_vice, gender=gender, pilot=pilot_version is not None
    )
    # Cut Path at VICE/
    param_path_vice = str(path_to_vice_sim_mat).split(f"VICE/{session}/")[-1]

    vice_weights = load_vice_weights(
        session=session,
        pilot=pilot_version is not None,
        pruned=True,
        return_path=False,
        param_path=param_path_vice,
    )[0]  # take only loc_params

    feat_tab = pd.DataFrame(index=list_of_models, columns=[f"D{i:03d}" for i in range(vice_weights.shape[1])])
    feat_tab.loc[:, :] = vice_weights

    # R(vice~BSM) of compute_vice_similarity_matrix > compute_vice_feature_map_similarity_matrix(..., z_score=True).
    # The Difference is that we do not z-score the features (dims) in the former case.
    # And we do not normalize the similarity matrix (0,1), however, this should have no effect on the R-value.
    # Z-scoring performs (probably) worse here, since we weaken the relevance of the sparse model dimensions,
    # in terms of its order.
    # Later dimensions are less important and represent stimulus similarities less well.
    # With z-scoring, we make differences between stimuli in these late dimensions as pronounced as in the first,
    # i.e., more relevant dimensions.
    # With z_score=False: == compute_vice_similarity_matrix,
    return compute_feature_similarity_matrix(feature_table=feat_tab, pca=pca, metric=metric, z_score=False)


def compute_vgg_feature_map_similarity_matrix(
    layer_name: str,
    pca: bool | float = False,
    data_mode: str = "3d-reconstructions",
    gender: str | None = None,
    pilot_version: int | None = None,
    metric: str = "cosine",
    extract_feat_maps: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Compute a similarity matrix from `VGGFace` feature maps.

    !!! note "To extend computational efficiency"
        Intermediate results are saved to disk, such that they do not have to be recomputed each time (time-consuming).

    :param pca: False OR provide (0.< pca < 1.) if PCA should be run on feature table with n components such that
                pca [float] *100 % of variance is explained
    :param layer_name: name of the VGG layer to use
    :param data_mode: the path to the "2d-original", "3d-reconstructions", or "3d-perspectives"
    :param gender: define gender if requested
    :param pilot_version: None for main experiment, OR pilot 1, OR 2
    :param metric: similarity metric to use ("cosine", "euclidean")
    :param extract_feat_maps: whether to extract feature maps from VGGFace
    :return: similarity matrix based on VGGFace feature maps
    """
    # Get head mapping table
    head_map = heads_naming_converter_table(pilot_version=pilot_version)

    # Prepare gender-only features if requested
    gender_suffix: str = ""
    if gender is not None:
        gender = check_gender(gender=gender)
        gender_suffix = f"_{gender}_only"

    # Set path to feature matrix
    data_mode = data_mode.lower()
    data_mode_suffix = "original" if "orig" in data_mode else "3D-recon" if "3d-recon" in data_mode else "3D-persp"
    p2_feat_mat = Path(
        paths.results.heads.vggface,
        f"VGGface_feature_maps_{data_mode_suffix}_{f'PCA-{pca:.2f}_' if pca else ''}{layer_name}.pd.pickle",
    )

    p2_feat_sim_mat = Path(str(p2_feat_mat).replace(".pd.pickle", f"_{metric}-similarity-matrix{gender_suffix}.npy"))
    p2_feat_mat.parent.mkdir(parents=True, exist_ok=True)  # create directory if not exists

    # Get the table with VGG-Face activations maps of all layers elicited by each head
    if (p2_feat_sim_mat.is_file() and extract_feat_maps) or not p2_feat_sim_mat.is_file():
        # optional when similarity matrix is already computed
        if p2_feat_mat.is_file():
            feat_tab = pd.read_pickle(p2_feat_mat)
        else:
            feat_tab = get_vgg_activation_maps(
                list_of_head_nrs=head_map.head_nr,
                layer_name=layer_name,
                data_mode=data_mode,
            )
            # we compute this for all heads irrespective of gender-only similarity matrices

            # Save feature table
            cprint(string="Saving feature table ...", col="b")
            feat_tab.to_pickle(p2_feat_mat)  # save feature table as pickle (pd.DataFrame) [fast & small]

    # Compute similarity matrix
    cprint(string=f"Computing similarity matrix for '{p2_feat_mat.name.split('.')[0]}' ...", col="b")
    if p2_feat_sim_mat.is_file():
        feat_sim_mat = np.load(file=p2_feat_sim_mat, allow_pickle=True)
        if pilot_version is not None:
            msg = "Cropping feat. sim. mat. for pilot is not implemented yet."
            raise NotImplementedError(msg)
    else:
        # Filter for gender if requested
        if gender is not None:
            feat_tab = feat_tab.loc[
                head_map[head_map.Model.str.contains("WF" if gender == "female" else "WM")].head_nr
            ]

        feat_sim_mat = compute_feature_similarity_matrix(feature_table=feat_tab, pca=pca, metric=metric, z_score=False)

        # Save similarity matrix
        cprint(string="Saving similarity matrix ...", col="b")
        np.save(file=p2_feat_sim_mat, arr=feat_sim_mat, allow_pickle=True)  # save similarity matrix
        # for z_scored is True: matrices are in "./results/heads/VGGface_zscored"

    return feat_sim_mat


def compute_vgg_human_judgment_feature_map_similarity_matrix(
    session: str,
    model_name: str | None = None,
    pca: bool | float = False,
    data_mode: str = "3d-reconstructions",
    gender: str | None = None,
    pilot_version: int | None = None,
    metric: str = "cosine",
    extract_feat_maps: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Compute a similarity matrix from feature maps of the `vgg_core_bridge` layer in `VGGFaceHumanjudgment[FrozenCore]`.

    !!! note "To extend computational efficiency"
        Intermediate results are saved to disk, such that they do not have to be recomputed each time (time-consuming).

    :param session: '2D', OR '3D'
    :param model_name: name of the model to use (if None, use the best model)
    :param pca: False OR provide (0.< pca < 1.) if PCA should be run on feature table with n components such that
                pca [float] *100 % of variance is explained
    :param data_mode: the path to the "2d-original", "3d-reconstructions", or "3d-perspectives"
    :param gender: define gender if requested
    :param pilot_version: None for main experiment, OR pilot 1, OR 2
    :param metric: similarity metric to use ("cosine", "euclidean")
    :param extract_feat_maps: whether to extract feature maps from VGGFace
    :return: similarity matrix based on VGGFace feature maps
    """
    # Get model information
    model_info = get_model_hps(
        session=session,
        model_name=model_name,
        data_mode=data_mode,
        exclusive_gender_trials=gender,
    )

    # Get head mapping table
    head_map = heads_naming_converter_table(pilot_version=pilot_version)

    # Prepare gender-only features if requested
    gender_suffix: str = ""
    if gender is not None:
        gender = check_gender(gender=gender)
        gender_suffix = f"_{gender}_only"

    # Set path to feature matrix
    data_mode_suffix = (
        "original"
        if "orig" in model_info.data_mode
        else "3D-recon"
        if "3d-recon" in model_info.data_mode
        else "3D-persp"
    )
    p2_feat_mat = Path(
        paths.results.main.vgg.feature_maps.format(session=session),
        f"{model_info.model_name}_feature_maps_{data_mode_suffix}_"
        f"{f'PCA-{pca:.2f}_' if pca else ''}vgg_core_bridge.pd.pickle",
    )
    p2_feat_sim_mat = Path(str(p2_feat_mat).replace(".pd.pickle", f"_{metric}-similarity-matrix{gender_suffix}.npy"))
    p2_feat_mat.parent.mkdir(parents=True, exist_ok=True)  # create directory if not exists

    # Get the table with VGGFaceHumanjudgment[FrozenCore] activations maps of all layers elicited by each head
    if not p2_feat_sim_mat.is_file() or extract_feat_maps:
        # optional when similarity matrix is already computed
        if p2_feat_mat.is_file():
            feat_tab = pd.read_pickle(p2_feat_mat)
        else:
            feat_tab = get_vgg_human_judgment_activation_maps(
                list_of_head_nrs=head_map.head_nr,
                session=session,
                model_name=model_info.model_name,
                data_mode=model_info.data_mode,
                exclusive_gender_trials=gender,
            )
            # compute this for all heads irrespective of gender-only similarity matrices

            # Save feature table
            cprint(string="Saving feature table ...", col="b")
            feat_tab.to_pickle(p2_feat_mat)  # save feature table as pickle (pd.DataFrame) [fast & small]

    # Compute similarity matrix
    cprint(string=f"Computing similarity matrix for '{p2_feat_mat.name.split('.')[0]}' ...", col="b")
    if p2_feat_sim_mat.is_file():
        feat_sim_mat = np.load(file=p2_feat_sim_mat, allow_pickle=True)
        if pilot_version is not None:
            msg = "Cropping feat. sim. mat. for pilot is not implemented yet."
            raise NotImplementedError(msg)
    else:
        # Filter for gender if requested
        if gender is not None:
            feat_tab = feat_tab.loc[
                head_map[head_map.Model.str.contains("WF" if gender == "female" else "WM")].head_nr
            ]

        feat_sim_mat = compute_feature_similarity_matrix(feature_table=feat_tab, pca=pca, metric=metric, z_score=False)

        # Save similarity matrix
        cprint(string="Saving similarity matrix ...", col="b")
        np.save(file=p2_feat_sim_mat, arr=feat_sim_mat, allow_pickle=True)  # save similarity matrix

    return feat_sim_mat


def visualise_matrix(
    face_sim_mat: np.ndarray,
    session: str,
    ppid: str | None = None,
    pilot: bool = params.PILOT,
    use_rsatoolbox: bool = False,
    save: bool = False,
    **kwargs,
) -> str | plt.Figure:
    """
    Visualize face similarity judgments.

    :param face_sim_mat: matrix of face similarity judgments of given participant
    :param ppid: ID of participant OR 'all'
    :param pilot: True: use pilot data
    :param session: '2D', OR '3D'
    :param use_rsatoolbox: plot with rsatoolbox
    :param save: save figure
    :return: None
    """
    # Plot matrix
    if ppid is None:
        fig_name = kwargs.pop("fig_name", f"Aggregated similarity judgments in {session}-session")
    else:
        fig_name = kwargs.pop("fig_name", f"Similarity judgments of PID {ppid} in {session}-session")

    # Compute size of the figure
    figsize = kwargs.pop(
        "figsize",
        (
            round(face_sim_mat.shape[1] / min(face_sim_mat.shape) * 10),  # keep x-axis longer since we add colorbar
            round(face_sim_mat.shape[0] / min(face_sim_mat.shape) * 9),
        ),
    )

    if use_rsatoolbox:
        # Explore rsatoolbox:
        #  data = rsatoolbox.data.Dataset(np.random.rand(10, 5))  # noqa: ERA001
        #  rdms = rsatoolbox.rdm.calc_rdm(data)  # noqa: ERA001
        # This is not ideal for our case, since it works with data with shape of (observations x channels).
        # With the following vectorized version, the visualization works ...
        rdms = rsatoolbox.rdm.RDMs(dissimilarities=vectorize_similarity_matrix(face_sim_mat=face_sim_mat))

        # TODO: Set labels (this is not functional yet)  # noqa: FIX002
        if "pattern_descriptor" in kwargs:
            rdms.pattern_descriptors.update(
                {"labels": heads_naming_converter_table(pilot_version=2 if pilot else None).head_nr.to_list()}
            )
            rdms.pattern_descriptors.update(
                {"index": np.arange(params.pilot.v2.n_faces if pilot else params.main.n_faces)}
            )

        fig, ax_array, _ = rsatoolbox.vis.show_rdm(
            rdm=rdms,
            show_colorbar="panel",
            vmin=kwargs.pop("vmin", 0.0),
            vmax=kwargs.pop("vmax", 1.0),
            figsize=figsize,
            rdm_descriptor=fig_name,
            num_pattern_groups=face_sim_mat.shape[0] / 2 if face_sim_mat.shape[0] % 2 == 0 else None,
            pattern_descriptor=kwargs.pop("pattern_descriptor", None),  # labels OR index
        )  # cmap="viridis"
        # plt.tight_layout()  # noqa: ERA001

        # Set labels and title
        msg = "Not implemented for rsatoolbox. Use its pattern_descriptor instead."
        if "xticklabels" in kwargs:
            raise NotImplementedError(msg)
        if "yticklabels" in kwargs:
            raise NotImplementedError(msg)
        if "xlabel" in kwargs:
            ax_array[0][0].set_xlabel(kwargs.pop("xlabel"))
        if "ylabel" in kwargs:
            ax_array[0][0].set_ylabel(kwargs.pop("ylabel"))
        if "title" in kwargs:
            ax_array[0][0].set_title(kwargs.pop("title"), pad=10)

    else:
        fig, ax1 = plt.subplots(num=fig_name, figsize=figsize, ncols=1)
        pos = ax1.imshow(face_sim_mat, cmap=kwargs.pop("cmap", None))
        # cmap='magma' OR 'inferno', interpolation='none')
        fig.colorbar(pos, ax=ax1)

        # Set labels and title
        if "xticklabels" in kwargs:
            ax1.set_xticks(range(len(kwargs["xticklabels"])))
            ax1.set_xticklabels(kwargs.pop("xticklabels"), rotation=45, ha="right", rotation_mode="anchor")
        if "yticklabels" in kwargs:
            ax1.set_yticks(range(len(kwargs["yticklabels"])))
            ax1.set_yticklabels(kwargs.pop("yticklabels"), rotation=45, ha="right", rotation_mode="anchor")
        if "xlabel" in kwargs:
            ax1.set_xlabel(kwargs.pop("xlabel"))
        if "ylabel" in kwargs:
            ax1.set_ylabel(kwargs.pop("ylabel"))
        if "title" in kwargs:
            ax1.set_title(kwargs.pop("title"), pad=10)

    if save:
        # Save figure
        p2save = Path(kwargs.pop("save_path", paths.results.pilot.v2.rdms if pilot else paths.results.main.rdms))
        p2save.mkdir(parents=True, exist_ok=True)
        for ext in ["png", "svg"]:
            p2_save_file = p2save / f"{fig_name}{'_rsatoolbox' if use_rsatoolbox else ''}.{ext}"
            cprint(string=f"Saving figure in {p2_save_file} ... ", col="b")
            plt.savefig(fname=p2_save_file, dpi=300, format=ext)
        plt.close()
        return str(p2_save_file)

    plt.show(block=False)
    return fig


def extract_set_of_heads(trial_results_table: pd.DataFrame) -> list:
    """
    Extract the set of heads from a trial-results table.

    That is, the heads that appeared in the experiment.
    """
    if "head1" in trial_results_table.columns:
        return sorted(np.unique(trial_results_table[["head1", "head2", "head3"]].to_numpy().flatten()))

    if "triplet" in trial_results_table.columns:
        heads = trial_results_table.triplet.apply(lambda x: x.split("_"))
        return sorted(np.unique(np.concatenate(heads.to_numpy()).astype(int)))

    msg = "Unknown trial results table format. 'head1', 'head2', 'head3' AND/OR 'triplet' are expected as columns."
    raise ValueError(msg)


def compute_similarity_matrix_from_human_judgments(
    trial_results_table: pd.DataFrame,
    pilot: bool = params.PILOT,
    split_return: bool = False,
    n_faces: int | None = None,
    multi_triplet_mode: str = "majority",
    verbose: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute a behavioral face similarity matrix (`BSM`) from the given a trial results table.

    :param trial_results_table: table with trial results
    :param pilot: True: use the pilot-data
    :param split_return: split the return in judgments and counts for aggregation across the participants
    :param n_faces: number of faces in the experiment
    :param multi_triplet_mode: how to handle trials with multiple samples
    :param verbose: be verbose
    :return: either aggregated matrix of similarity judgments OR split in judgments and counts
    """
    # Remove training trials
    trial_results_table = trial_results_table.sort_values(
        by=list(set(trial_results_table.columns).intersection(["ppid_session_dataname", "trial_num"]))
    ).reset_index(drop=True)

    # For triplet-IDs with multiple samples, we should take the majority vote or sample randomly
    tr_tab_tripl_val_ctn = trial_results_table.triplet_id.value_counts()
    if tr_tab_tripl_val_ctn.nunique() > 1:  # noqa: PD101
        multi_triplet_mode = multi_triplet_mode.lower()
        if multi_triplet_mode not in {"majority", "random", "ignore"}:
            msg = "multi_triplet_mode must be 'majority' OR 'random'!"
            raise ValueError(msg)

        min_n_samples = tr_tab_tripl_val_ctn.min()
        if multi_triplet_mode == "majority":
            # Per triplet ID keep only the majority vote
            for t_id in tqdm(
                tr_tab_tripl_val_ctn[tr_tab_tripl_val_ctn > min_n_samples].index,
                desc="Clean table from multiple samples of triplet IDs via majority vote.",
                total=(tr_tab_tripl_val_ctn > min_n_samples).sum(),
            ):
                # Head odd counts
                head_odd_counts = trial_results_table[trial_results_table.triplet_id == t_id].head_odd.value_counts()

                # Find head which was chosen most often
                head_odd_majority = head_odd_counts[head_odd_counts == head_odd_counts.max()].sample(1).index[0]
                # we sample here, since there might be multiple odd heads with the same count (i.e., max)

                # Find indices of the current triplet-ID and the majority head
                indices_to_kick = list(
                    trial_results_table.loc[
                        (
                            (trial_results_table.triplet_id == t_id)
                            & (trial_results_table.head_odd == head_odd_majority)
                        )
                    ].index
                )
                # Kick one random index to keep it in the table
                np.random.shuffle(indices_to_kick)
                indices_to_kick.pop()
                # Add the indices of the other heads to kick
                indices_to_kick += list(
                    trial_results_table.loc[
                        (
                            (trial_results_table.triplet_id == t_id)
                            & (trial_results_table.head_odd != head_odd_majority)
                        )
                    ].index
                )

                trial_results_table = trial_results_table.drop(index=indices_to_kick)

        elif multi_triplet_mode == "random":
            trial_results_table = trial_results_table.groupby("triplet_id").sample(n=min_n_samples)

        else:  # multi_triplet_mode == "ignore"
            val_ctn_more_than_min = tr_tab_tripl_val_ctn[tr_tab_tripl_val_ctn > min_n_samples]
            if verbose:
                cprint(
                    string=f"Ignoring {sum(val_ctn_more_than_min)} multiple samples of "
                    f"{len(val_ctn_more_than_min)} triplet IDs!",
                    col="y",
                )

    if n_faces is None:
        n_faces = params.pilot.v2.n_faces if pilot else params.main.n_faces  # 25 or 100
    list_of_heads_in_table = extract_set_of_heads(trial_results_table)

    prev_indexing = False
    if len(list_of_heads_in_table) != n_faces:
        cprint(
            string=f"Not all {n_faces} faces are present in the trial data.\n"
            f"Consider passing n_faces={len(list_of_heads_in_table)} as kwarg!",
            col="r",
        )
        prev_indexing = True

    face_sim_mat = np.identity(n_faces)
    face_ctn_mat = np.zeros(shape=(n_faces, n_faces))

    # Extract data
    # rt = tab["response_time"]  # keep for potential later usage ...  # noqa: ERA001
    judge = trial_results_table[["head1", "head2", "head3", "head_odd"]]

    nan_trials = 0
    miss_trials = 0
    for _row_i, (h1, h2, h3, odd) in judge.iterrows():  # row_i is not used here
        if np.isnan((h1, h2, h3, odd)).any():
            nan_trials += 1
            continue

        if odd == 0:
            miss_trials += 1
            continue

        # Determine indices of face-pairs in the similarity matrix
        for heads_combi in itertools.combinations((h1, h2, h3), r=2):
            h_i, h_ii = heads_combi  # save which heads are in combo

            if prev_indexing:
                indices = np.array(heads_combi).astype(int)

                if pilot:
                    # In pilot (v2) female faces have head number 1 to 12, and male faces 51 to 63,
                    # we want to map this to the indices female: 0-11, male: 12-24
                    for i, fidx in enumerate(indices):
                        indices[i] = head_nr_to_pilot_matrix_index(head_id=fidx, pilot_version=2)
                else:  # main experiment
                    indices -= 1

                indices = tuple(indices)
            else:
                indices = list_of_heads_in_table.index(h_i), list_of_heads_in_table.index(h_ii)  # tuple

            # Count comparisons
            face_ctn_mat[indices] += 1
            face_ctn_mat[indices[::-1]] += 1  # fill symmetrically

            # Fill judgments
            similar = int(odd not in {h_i, h_ii})
            face_sim_mat[indices] += similar
            face_sim_mat[indices[::-1]] += similar  # fill symmetrically

    if split_return:
        # For aggregation across participants
        return face_sim_mat, face_ctn_mat

    # Average across trials
    face_ctn_mat[np.where(face_ctn_mat == 0)] = np.nan
    return face_sim_mat / face_ctn_mat


def compute_similarity_matrix_from_vgg_face_human_judgment_model(
    session: str,
    model_name: str | None = None,
    split_return: bool = False,
    n_faces: int | None = None,
    exclusive_gender_trials: str | None = None,
    verbose: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute a face similarity matrix from decisions of the `VGGFaceHumanjudgment[FrozenCore]` model.

    :param session: '2D', OR '3D'
    :param model_name: name of the model to use
    :param split_return: split the return in the decisions, and the count
    :param n_faces: the number of faces in the experiment
    :param exclusive_gender_trials: use exclusive gender trials ['female' OR 'male'], OR None for all samples.
    :param verbose: be verbose
    :return: either matrix of similarity decisions OR split in decisions and counts
    """
    # Check input
    session = session.upper()
    if session not in params.SESSIONS:
        msg = f"session must be in {params.SESSIONS}"
        raise ValueError(msg)

    # Get model information
    vgg_hj_perform_info = get_model_hps(
        session=session, model_name=model_name, exclusive_gender_trials=exclusive_gender_trials
    )
    vgg_hj_model_name = vgg_hj_perform_info.model_name
    vgg_hj_data_mode = vgg_hj_perform_info.data_mode

    # Set path to decision table
    g_sfx = "" if exclusive_gender_trials is None else f"_{exclusive_gender_trials.lower()}_only"

    p2_model_decisions = Path(paths.results.main.VGG, session, f"{vgg_hj_model_name}{g_sfx}_decisions").with_suffix(
        ".csv"
    )

    # Get / compute decision table for all participants
    if p2_model_decisions.exists():
        trial_results_table = pd.read_csv(p2_model_decisions)

    else:
        import torch

        # Get model
        vgg_hj_model = load_trained_vgg_face_human_judgment_model(
            session=session,
            model_name=vgg_hj_model_name,
            exclusive_gender_trials=exclusive_gender_trials,
            device="gpu" if torch.cuda.is_available() else "cpu",
        )
        vgg_hj_model.eval()

        # Get model data
        full_dataset_dl, _, _ = prepare_data_for_human_judgment_model(
            session=session,
            frozen_core=vgg_hj_model.freeze_vgg_core,
            data_mode=vgg_hj_data_mode,
            last_core_layer=vgg_hj_model.last_core_layer,
            split_ratio=(1.0, 0.0, 0.0),  # push all data in one set
            batch_size=1,
            num_workers=1,
            dtype=torch.float32,
            exclusive_gender_trials=exclusive_gender_trials,
        )

        # Init table
        trial_results_table = pd.DataFrame(
            columns=["head1", "head2", "head3", "head_odd_human_choice", "head_odd_model_choice"]
        )

        # Fill table with model decisions
        with torch.no_grad():
            for i, model_input in tqdm(
                enumerate(full_dataset_dl),
                desc=f"Get decisions of '{vgg_hj_model_name}'",
                total=len(full_dataset_dl),
                colour="#57965D",
            ):
                ipt1, ipt2, ipt3, _, idx = model_input.values()  # _ == choice
                i_decision = vgg_hj_model(ipt1, ipt2, ipt3).argmax().item()

                trial_results_table.loc[i, :] = (
                    full_dataset_dl.dataset.dataset.session_data.iloc[idx.item()].to_list()  # noqa: RUF005
                    + [None]
                )
                trial_results_table.loc[i, "head_odd_model_choice"] = trial_results_table.loc[
                    i, ["head1", "head2", "head3"]
                ][i_decision]

        # Save decision_table
        p2_model_decisions.parent.mkdir(parents=True, exist_ok=True)
        trial_results_table.to_csv(p2_model_decisions, index=False)

    if verbose:
        perc_match = (trial_results_table.head_odd_human_choice == trial_results_table.head_odd_model_choice).mean()
        cprint(string=f"\n\t{perc_match:.2%} of '{vgg_hj_model_name}' decisions match human judgements.\n", col="g")

    # Determine the number of faces
    if n_faces is None:
        n_faces = params.main.n_faces
    list_of_heads_in_table = extract_set_of_heads(trial_results_table)

    prev_indexing = False
    if len(list_of_heads_in_table) != n_faces:
        cprint(
            string=f"Not all {n_faces} faces are present in the trial data.\n"
            f"Consider passing n_faces={len(list_of_heads_in_table)} as kwarg!",
            col="r",
        )
        prev_indexing = True

    # Init similarity matrix
    face_sim_mat = np.identity(n_faces)
    face_ctn_mat = np.zeros(shape=(n_faces, n_faces))

    # Extract data
    judge = trial_results_table[["head1", "head2", "head3", "head_odd_model_choice"]]
    for _, (h1, h2, h3, odd) in tqdm(
        judge.iterrows(),
        desc=f"Compute similarity matrix from decisions of '{vgg_hj_model_name}'",
        total=len(judge),
        colour="#F98382",
    ):
        # Determine indices of face-pairs in the similarity matrix
        for heads_combi in itertools.combinations((h1, h2, h3), r=2):
            h_i, h_ii = heads_combi  # save which heads are in combo

            if prev_indexing:
                indices = np.array(heads_combi).astype(int)
                indices -= 1  # 1 == judge.min().min()
                indices = tuple(indices)
            else:
                indices = list_of_heads_in_table.index(h_i), list_of_heads_in_table.index(h_ii)  # tuple

            # Count comparisons
            face_ctn_mat[indices] += 1
            face_ctn_mat[indices[::-1]] += 1  # fill symmetrically

            # Fill judgments
            similar = int(odd not in {h_i, h_ii})
            face_sim_mat[indices] += similar
            face_sim_mat[indices[::-1]] += similar  # fill symmetrically

    if split_return:
        # For aggregation across participants
        return face_sim_mat, face_ctn_mat

    # Average across trials
    face_ctn_mat[np.where(face_ctn_mat == 0)] = np.nan
    return face_sim_mat / face_ctn_mat


def similarity_judgments_of_single_participant(
    ppid: str, pilot: bool = params.PILOT, split_return: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute a face similarity matrix from a single participant's behavioral data.

    :param ppid: ID of participant
    :param pilot: True: use the pilot-data
    :param split_return: split the return in judgments and counts for aggregation across participants
    :return: either aggregated matrix of similarity judgments OR split in judgments and counts
    """
    if pilot:
        tab = read_pilot_data(clean_trials=True, verbose=False)
        tab = tab.loc[tab.ppid == ppid]  # reduce table to given PID
    else:
        tab = read_trial_results_of_participant(ppid=ppid, clean_trials=True, verbose=False)

    return compute_similarity_matrix_from_human_judgments(
        trial_results_table=tab, pilot=pilot, split_return=split_return
    )


def extract_exclusive_gender_trials(
    trial_results_table: pd.DataFrame, gender: str, verbose: bool = False
) -> pd.DataFrame:
    """Extract triplets that contain only the faces of one gender."""
    gender = check_gender(gender)

    # Define drop condition for index
    # (could be done for both genders at once ...)
    n_all_trials = len(trial_results_table)
    n_one_gender = 50
    drop_cond = (lambda x: (x > n_one_gender).any()) if gender == "female" else (lambda x: (x <= n_one_gender).any())
    drop_indices = []  # init list for indices to drop
    for i, tr_row in tqdm(
        trial_results_table[["head1", "head2", "head3"]].iterrows(),
        desc=f"Filter table for {gender} only trials",
        total=n_all_trials,
    ):
        if drop_cond(tr_row):
            drop_indices.append(i)
    trial_results_table = trial_results_table.drop(index=drop_indices, inplace=False).reset_index(drop=True)
    if verbose:
        cprint(string=f"From initial {n_all_trials} trials, {len(trial_results_table)} trials remain.", col="b")

    return trial_results_table


def aggregate_judgments_in_session(
    session: str, pilot: bool = params.PILOT, recalculate: bool = False, verbose: bool = False
) -> np.ndarray:
    """
    Aggregate similarity judgments across participants in the given session (2D, 3D).

    :param session: '2D', OR '3D'
    :param pilot: True: use pilot data
    :param recalculate: do not use cached table, recalculate similarity judgments instead
    :param verbose: verbose or not
    :return: matrix with normalized judgments
    """
    # Define path to cached similarity judgments matrix
    path_to_cached_matrix = Path(
        paths.results.pilot.v2.rsa if pilot else paths.results.main.rsa,
        f"cached_{session}_similarity_judgments_matrix.npy",
    )

    # Check, if cached similarity judgments matrix exists, and if it should be recalculated
    load_cached_matrix: bool = path_to_cached_matrix.is_file() and not recalculate

    if load_cached_matrix:
        if verbose:
            cprint(string=f"Loading cached similarity judgments matrix from {path_to_cached_matrix} ... ", col="b")
        sim_mat: np.ndarray = np.load(path_to_cached_matrix)

    else:
        if pilot:
            tab = read_pilot_data(clean_trials=True, verbose=verbose)
        else:
            tab = read_trial_results_of_session(
                session=session, clean_trials=True, drop_subsamples=True, verbose=verbose
            )

        sim_mat: np.ndarray = compute_similarity_matrix_from_human_judgments(
            trial_results_table=tab, pilot=pilot, split_return=False
        )

        if verbose:
            cprint(string=f"Saving similarity judgments matrix to {path_to_cached_matrix} ... ", col="b")
        np.save(path_to_cached_matrix, sim_mat)

    return sim_mat


@lru_cache
def aggregate_judgments_in_session_by_gender(
    session: str, gender: str, pilot: bool = params.PILOT, recalculate: bool = False, verbose: bool = False
) -> np.ndarray:
    """
    Aggregate similarity judgments across participants in the given session (2D, 3D).

    Take only trials of triplets that exclusively contain the faces of the given gender.

    :param session: '2D', OR '3D'
    :param gender: 'female', OR 'male'
    :param pilot: True: use pilot data
    :param recalculate: do not use cached table, recalculate similarity judgments instead
    :param verbose: verbose or not
    :return: matrix with normalized judgments
    """
    gender = check_gender(gender=gender)

    # Define path to cached similarity judgments matrix
    path_to_cached_matrix = Path(
        paths.results.pilot.v2.rsa if pilot else paths.results.main.rsa,
        f"cached_{session}_{gender}_similarity_judgments_matrix.npy",
    )

    # Check, if cached similarity judgments matrix exists, and if it should be recalculated
    load_cached_matrix: bool = path_to_cached_matrix.is_file() and not recalculate

    if load_cached_matrix:
        if verbose:
            cprint(string=f"Loading cached similarity judgments matrix from {path_to_cached_matrix} ... ", col="b")
        sim_mat: np.ndarray = np.load(path_to_cached_matrix)
    else:
        if pilot:
            tab = read_pilot_data(clean_trials=True, verbose=verbose)
            msg = "Not implemented for pilot data."
            raise NotImplementedError(msg)
        else:  # noqa: RET506
            tab = read_trial_results_of_session(
                session=session, clean_trials=True, drop_subsamples=True, verbose=verbose
            )
            tab = extract_exclusive_gender_trials(trial_results_table=tab, gender=gender, verbose=verbose)
        sim_mat = compute_similarity_matrix_from_human_judgments(
            trial_results_table=tab, pilot=pilot, split_return=False
        )

        # Reduce matrix to gender of interest
        n_gender = (12 if gender == "female" else 13) if pilot else (params.main.n_faces // 2)
        sim_mat = sim_mat[:n_gender, :n_gender] if gender == "female" else sim_mat[n_gender:, n_gender:]  # or "male"

        if verbose:
            cprint(string=f"Saving similarity judgments matrix to {path_to_cached_matrix} ... ", col="b")
        np.save(path_to_cached_matrix, sim_mat)

    return sim_mat


def vectorize_similarity_matrix(face_sim_mat: np.ndarray) -> np.ndarray:
    """
    Take the upper triangle of a given similarity matrix and return it as vector.

    ??? note "Ways to vectorize a matrix"
        ```python
            a = [[1 2 3]
                 [4 5 6]
                 [7 8 9]]

            # This is how the diagonal can be excluded (as it is required here)
            print(a[np.triu_indices(n=3, k=1)])
            # > array([2, 3, 6])

            # In contrast, see how the diagonal can be included (as it is not done here):
            print(a[np.triu_indices(n=3, k=0)])
            # > array([1, 2, 3, 5, 6, 9])
        ```

    :param face_sim_mat: face similarity matrix
    :return: 1d vector of upper triangle
    """
    return face_sim_mat[np.triu_indices(n=face_sim_mat.shape[0], k=1)]


def get_corr_df_rsm(corr_name: str, metric: str) -> pd.DataFrame:
    """
    Get the correlation dataframe for representational similarity matrices (`RSM`).

    :param corr_name: name of correlation to use ("Pearson", "Spearman")
    :param metric: similarity metric to use ("cosine", "euclidean")
    """
    return pd.read_csv(Path(paths.results.main.rsa, f"{corr_name.title()}_{metric.lower()}.csv"), index_col=0)


def plot_rsa_corr_df(corr_name: str, metric: str, corr_df: pd.DataFrame | None, save: bool = False, **kwargs) -> None:
    """
    Plot the RSA correlation dataframe.

    :param corr_name: name of correlation to use ("Pearson", "Spearman")
    :param metric: similarity metric to use ("cosine", "euclidean")
    :param corr_df: RSA correlation dataframe
    :param save: whether to save the plot
    """
    if corr_df is None:
        corr_df = get_corr_df_rsm(corr_name=corr_name, metric=metric)

    # Get max empirical R
    mer = (
        pd.read_csv(
            paths.results.main.noise_ceiling.r_table.format(corr_name=corr_name.lower()),
            index_col=["session", "sample_type"],
        )
        .loc[("both", "multi-sub-sample")]
        .mean_r
    )

    # Plot correlations between BSMs and physical / computational face features
    additional_sim_mat = kwargs.pop("additional_sim_mat", False)
    for bsm_name, corr_row in corr_df.iterrows():
        if not additional_sim_mat and str(bsm_name).endswith("_sim_mat"):
            continue
        sess = str(bsm_name).split("_")[0]
        other_sess = [s for s in params.SESSIONS if s != sess].pop()
        exclusive = "only" in bsm_name
        gender = "female" if "female" in bsm_name else "male" if "_male" in bsm_name else None
        tmp_corr_row = corr_row.dropna()

        # Filter columns / variables to plot
        #   Remove PCA columns & SPoSE columns
        r_cols = [c for c in tmp_corr_row.index if c.endswith("_r") and "PCA" not in c and "SPoSE" not in c]

        #   Remove other-session columns but not other BSM
        r_cols = [c for c in r_cols if "BSM" in c or f"_{other_sess}_" not in c]

        #   Filter for exclusive gender trials if required
        r_cols = [c for c in r_cols if "only" in c] if exclusive else [c for c in r_cols if "only" not in c]
        if exclusive:
            r_cols = [c for c in r_cols if f"_{gender}_" in c]

        #   Filter for VGG columns based on 3D-reconstructions
        r_cols = [c for c in r_cols if "VGG_org" not in c]

        #   Find three VGG columns with the highest correlation
        top_3_vgg = tmp_corr_row[[c for c in r_cols if c.startswith("VGG_")]].sort_values(ascending=False)[:3]
        #   Remove other VGG columns from r_cols
        r_cols = [c for c in r_cols if (not c.startswith("VGG_") or c in top_3_vgg)]

        #   Separate VGGFaceHumanjudgmentFrozenCore columns
        r_vgghj_cols = [c for c in r_cols if "VGGFaceHumanjudgment" in c]
        r_vgghj_cols_replace = ["VGG HJ" + c.split("_VGGFaceHumanjudgmentFrozenCore")[1] for c in r_vgghj_cols]

        plt.figure(figsize=(10, 8))
        h = tmp_corr_row[r_cols].plot(
            kind="bar",
            title=bsm_name,
            color=["#081C22"]  # BSM
            + ["#1E7872"]  # CFD PFF
            + ["#F4C096"] * 6  # DECA
            + ["#EE2E33"] * 3  # VGG-Face
            + ["#6C4179"] * 1  # VICE
            + ["#008080"] * 2,  # VGGFaceHumanjudgmentFrozenCore
        )

        h.set_xticklabels(
            labels=[
                c.replace(r_vgghj_cols[0], r_vgghj_cols_replace[0])
                .replace(r_vgghj_cols[1], r_vgghj_cols_replace[1])
                .removesuffix("_r")
                .replace("_", " ")
                .replace(" EXP", " EXPRESSION")
                .replace(" CAM", " CAMERA")
                .replace(" TEX", " TEXTURE")
                .replace("3D-recon", "")
                .replace(" inner", "")  # in case of PLOT_VICE_INNER
                for c in r_cols
            ],
            rotation=30,
            ha="right",
            fontdict={"size": 12},
        )
        ticks_loc = h.get_yticks().tolist()
        h.yaxis.set_major_locator(FixedLocator(ticks_loc))
        h.set_yticklabels(labels=[f"{t:.1f}" for t in h.get_yticks()], fontdict={"size": 12})
        h.set_ylim(0, 1)
        h.set_ylabel("Correlation R", fontdict={"size": 14})
        h.set_title(
            f"Correlation between {str(bsm_name).replace('_', ' ')} & other face features", fontdict={"size": 16}
        )
        # Add horizontal line for max empirical R
        h.axhline(y=mer, color="r", linestyle="--", alpha=0.5)  # , label="Max empirical R"
        plt.tight_layout()

        # Save figure
        if save:
            for ext in [".png", ".svg"]:
                plt.savefig(
                    Path(paths.results.main.rsa, f"{corr_name.title()}_{bsm_name}-FaceFeats_{metric}").with_suffix(ext)
                )
            plt.close()
        else:
            plt.show()


def plot_vgg_correlations(
    corr_name: str, metric: str, data_mode: str, max_r: float | None = None, save: bool = True
) -> None:
    """
    Plot correlations between similarity matrices.

    :param corr_name: name of correlation to use ("Pearson", "Spearman")
    :param metric: similarity metric to use ("cosine", "euclidean")
    :param data_mode: data mode ("2d-original", "3d-reconstructions", "3d-perspectives")
    :param max_r: maximum correlation value for limit of y-axis
    :param save: whether to save the plot
    """
    corr_name = corr_name.lower()
    metric = metric.lower()
    data_mode = data_mode.lower()
    if "3d-persp" in data_mode:
        msg = "3D-persp data mode is not implemented yet."
        raise NotImplementedError(msg)

    corr_df = get_corr_df_rsm(corr_name=corr_name, metric=metric)

    if max_r is None:
        max_r = corr_df.loc[:, [c for c in corr_df.columns if "_p" not in c and "VGG_" in c]].max().max()

    # Filter VGG columns
    ipt = "org_" if "original" in data_mode else "3D-recon_" if "3d-recon" in data_mode else "3D-persp_"
    non_ipt = "3D-recon_" if "original" in data_mode else "org_"
    vgg_cols = [c for c in corr_df.columns if ("VGG_" in c and "PCA" not in c and "_p" not in c and non_ipt not in c)]

    for bsm_name, other_sim_mat in corr_df.iterrows():
        if str(bsm_name).endswith("D_BSM"):
            # All trials (i.e., no gender-only trials)
            tmp_vgg_cols = [c for c in vgg_cols if "male_" not in c]
            gender_filter = ""
        else:
            gender_filter = "_" + str(bsm_name).split("BSM_")[-1]
            gender_filter += "_only" if not gender_filter.endswith("_only") else ""
            tmp_vgg_cols = [c for c in vgg_cols if gender_filter in c]

        sub_df = other_sim_mat[tmp_vgg_cols].copy()

        fig, ax = plt.subplots(figsize=(12, 8))
        h = sns.barplot(
            x=tmp_vgg_cols,
            y=sub_df.values,
            ax=ax,
            palette=sns.color_palette(palette="flare", as_cmap=False, n_colors=len(tmp_vgg_cols)),  # "dark:#5A9_r"
        )

        h.set_xticklabels(
            labels=[c.removeprefix(f"VGG_{ipt}").removesuffix(f"{gender_filter}_r") for c in tmp_vgg_cols],
            rotation=45,
            ha="right",
        )
        h.set_ylim(0, max_r)
        h.set_ylabel("Correlation R", fontsize=14)
        h.set_xlabel("VGGFace layers", fontsize=14)
        h.set_title(f"Correlation between {bsm_name} and VGGFace feature maps (input: {ipt.removesuffix('_')})")
        fig.tight_layout()

        if save:
            for ext in [".png", ".svg"]:
                fig.savefig(
                    Path(
                        paths.results.main.rsa,
                        f"VGG_{ipt.removesuffix('_')}_feat-{metric}-sim_{bsm_name}_{corr_name.lower()}_corr",
                    ).with_suffix(ext)
                )
            plt.close()


def main() -> None:
    """
    Run the main function of the `rsa.py` script.

    This script computes the correlation between the similarity judgments of the 2D and 3D sessions.
    Moreover, it runs RSA on different similarity matrices and creates plots.
    """
    # Set correlation function name
    corr_name = "Spearman" if FLAGS.spearman else "Pearson"  # OR: corr_func.__name__[:-1].title()

    # Set logger
    logger = logging.getLogger(__name__)  # get predefined logger
    logger_filename = (
        Path(paths.results.pilot.v2.rsa if FLAGS.pilot else paths.results.main.rsa)
        / f"logs/rsa_{corr_name}_{FLAGS.metric}.log"
    )
    if FLAGS.logger_overwrite and logger_filename.is_file():
        logger_filename.unlink()

    logger = update_logger_configs(new_logger_name="RSA", new_logger_filename=logger_filename, logger=logger)

    # %% Init correlation table (RSA)
    rsa_corr_df = pd.DataFrame(
        index=[sess + "_BSM" for sess in params.SESSIONS]
        + [sess + "_BSM_" + g for g in params.GENDERS for sess in params.SESSIONS]  # gender slices but mixed trials
        + [sess + "_BSM_" + f"{g}_only" for g in params.GENDERS for sess in params.SESSIONS]
        # exclusive gender trials
    )

    # %% Compute aggregated similarity judgments (behavioral similarity matrix, BSM)
    sim_mat_all_2d = aggregate_judgments_in_session(session="2D", pilot=FLAGS.pilot, verbose=FLAGS.verbose)
    sim_mat_all_3d = aggregate_judgments_in_session(session="3D", pilot=FLAGS.pilot, verbose=FLAGS.verbose)

    # %% Compute BSMs with exclusive gender trials
    sim_mat_all_2d_female = aggregate_judgments_in_session_by_gender(
        session="2D", gender="female", pilot=FLAGS.pilot, verbose=FLAGS.verbose
    )
    sim_mat_all_2d_male = aggregate_judgments_in_session_by_gender(
        session="2D", gender="male", pilot=FLAGS.pilot, verbose=FLAGS.verbose
    )

    sim_mat_all_3d_female = aggregate_judgments_in_session_by_gender(
        session="3D", gender="female", pilot=FLAGS.pilot, verbose=FLAGS.verbose
    )
    sim_mat_all_3d_male = aggregate_judgments_in_session_by_gender(
        session="3D", gender="male", pilot=FLAGS.pilot, verbose=FLAGS.verbose
    )

    sim_mats_by_exclusive_gender_dict = {
        "2D": {"female": sim_mat_all_2d_female, "male": sim_mat_all_2d_male},
        "3D": {"female": sim_mat_all_3d_female, "male": sim_mat_all_3d_male},
    }
    # Note: These contain per-gender-only-triplets which exclusively consist of the respective gender

    # %% Visualize the BSMs
    n_female: int = 12 if FLAGS.pilot else 50
    if FLAGS.plot:
        for session, sim_mat_all in zip(params.SESSIONS, [sim_mat_all_2d, sim_mat_all_3d], strict=True):
            visualise_matrix(
                face_sim_mat=sim_mat_all,
                session=session,
                pilot=FLAGS.pilot,
                use_rsatoolbox=FLAGS.rsa_toolbox,
                save=FLAGS.save_plots,
            )
            # Plot for each gender
            for gender, slice_it in zip(
                params.GENDERS, [slice(0, n_female, 1), slice(n_female, None, 1)], strict=True
            ):
                visualise_matrix(
                    face_sim_mat=sim_mat_all[slice_it, slice_it],
                    session=session,
                    pilot=FLAGS.pilot,
                    fig_name=f"{gender}_{session}",
                    use_rsatoolbox=FLAGS.rsa_toolbox,
                    save=FLAGS.save_plots,
                )

        # Plot exclusive gender trials
        for session in params.SESSIONS:
            for gender in params.GENDERS:
                visualise_matrix(
                    face_sim_mat=sim_mats_by_exclusive_gender_dict[session][gender],
                    session=session,
                    pilot=FLAGS.pilot,
                    use_rsatoolbox=FLAGS.rsa_toolbox,
                    fig_name=f"{gender}-only_{session}",
                    save=FLAGS.save_plots,
                )

    # %% Compute the correlation between similarity judgments (BSMs) of both conditions (2D & 3D)
    corr_func = spearmanr if FLAGS.spearman else pearsonr

    r, p = corr_func(vectorize_similarity_matrix(sim_mat_all_2d), vectorize_similarity_matrix(sim_mat_all_3d))
    log_msg = (
        f"{corr_name} correlation between similarity judgments of 2D & 3D: R={r:.3f}, p<={p:.5g}."
        f"\t{1 - r**2:.2%} of variance in one condition remains unexplained by the other."
    )
    logger.info(msg=log_msg)

    # Save results
    rsa_corr_df.loc["2D_BSM", "3D_BSM_r"] = r
    rsa_corr_df.loc["2D_BSM", "3D_BSM_p"] = p
    rsa_corr_df.loc["3D_BSM", "2D_BSM_r"] = r
    rsa_corr_df.loc["3D_BSM", "2D_BSM_p"] = p

    # Compute correlations of BSMs for exclusive and non-exclusive gender trials
    for gender, slice_it in zip(params.GENDERS, [slice(0, n_female, 1), slice(n_female, None, 1)], strict=True):
        # Compute for non-exclusive gender trials
        r, p = corr_func(
            vectorize_similarity_matrix(sim_mat_all_2d[slice_it, slice_it]),
            vectorize_similarity_matrix(sim_mat_all_3d[slice_it, slice_it]),
        )
        log_msg = (
            f"{corr_name} correlation between similarity judgments of 2D & 3D within {gender} (non-exclusive): "
            f"r={r:.3f}, p<={p:.5g}."
            f"\t{1 - r**2:.2%} of variance in one condition remains unexplained by the other."
        )
        logger.info(msg=log_msg)

        rsa_corr_df.loc[f"2D_BSM_{gender}", f"3D_BSM_{gender}_r"] = r
        rsa_corr_df.loc[f"2D_BSM_{gender}", f"3D_BSM_{gender}_p"] = p
        rsa_corr_df.loc[f"3D_BSM_{gender}", f"2D_BSM_{gender}_r"] = r
        rsa_corr_df.loc[f"3D_BSM_{gender}", f"2D_BSM_{gender}_p"] = p

        # Compute for exclusive gender trials
        r, p = corr_func(
            vectorize_similarity_matrix(sim_mats_by_exclusive_gender_dict["2D"][gender]),
            vectorize_similarity_matrix(sim_mats_by_exclusive_gender_dict["3D"][gender]),
        )

        log_msg = (
            f"{corr_name} correlation between similarity judgments of 2D & 3D within {gender} only: "
            f"r={r:.3f}, p<={p:.5g}."
            f"\t{1 - r**2:.2%} of variance in one condition remains unexplained by the other."
        )
        logger.info(msg=log_msg)

        rsa_corr_df.loc[f"2D_BSM_{gender}_only", f"3D_BSM_{gender}_only_r"] = r
        rsa_corr_df.loc[f"2D_BSM_{gender}_only", f"3D_BSM_{gender}_only_p"] = p
        rsa_corr_df.loc[f"3D_BSM_{gender}_only", f"2D_BSM_{gender}_only_r"] = r
        rsa_corr_df.loc[f"3D_BSM_{gender}_only", f"2D_BSM_{gender}_only_p"] = p

    # Sanity check: correlation between exclusive gender trials (female ~ male; should be low -> 0)
    for (sess_1, gender_1), (sess_2, gender_2) in itertools.combinations(
        itertools.product(params.SESSIONS, params.GENDERS), r=2
    ):
        if gender_1 == gender_2:
            # Do not compare within gender across sessions (e.g., "male-2D" vs "male-3D"), since this is done elsewhere
            continue

        r, p = corr_func(
            vectorize_similarity_matrix(sim_mats_by_exclusive_gender_dict[sess_1][gender_1]),
            vectorize_similarity_matrix(sim_mats_by_exclusive_gender_dict[sess_2][gender_2]),
        )
        log_msg = (
            f"{corr_name} correlation between similarity judgments of {sess_1} & {sess_2} between gender-only "
            f"({gender_1}~{gender_2}): r={r:.3f}, p<={p:.5g}."
            f"\t{1 - r**2:.2%} of variance in one condition remains unexplained by the other."
        )
        logger.info(msg=log_msg)

        # Save results
        rsa_corr_df.loc[f"{sess_1}_BSM_{gender_1}_only", f"{sess_2}_BSM_{gender_2}_only_r"] = r
        rsa_corr_df.loc[f"{sess_1}_BSM_{gender_1}_only", f"{sess_2}_BSM_{gender_2}_only_p"] = p
        rsa_corr_df.loc[f"{sess_2}_BSM_{gender_2}_only", f"{sess_1}_BSM_{gender_1}_only_r"] = r
        rsa_corr_df.loc[f"{sess_2}_BSM_{gender_2}_only", f"{sess_1}_BSM_{gender_1}_only_p"] = p

    # %% Compute (cosine/euclidean) similarity of (physical and computational) face features, also with PCA version
    # Compute similarity matrices based on CFD physical face features (PFF)
    feature_dict = {"CFD_PFF": compute_physical_attr_similarity_matrix}
    # Compute for exclusive gender trials
    for gender in params.GENDERS:
        feature_dict[f"CFD_PFF_{gender}_only"] = partial(compute_physical_attr_similarity_matrix, gender=gender)

    # Compute similarity matrices based on FLAME/DECA features
    list_of_flame_params = latent_flame_code + (["detail"] if FLAGS.flame_model == "deca" else [])
    for feat in list_of_flame_params:  # add FLAME features
        feature_dict[f"{FLAGS.flame_model.upper()}_{feat.upper()}"] = partial(
            compute_flame_feature_similarity_matrix,
            param=feat,
            model=FLAGS.flame_model,
            pilot_version=2 if FLAGS.pilot else None,
        )
        # Compute for within gender stimuli
        for gender in params.GENDERS:
            feature_dict[f"{FLAGS.flame_model.upper()}_{feat.upper()}_{gender}_only"] = partial(
                compute_flame_feature_similarity_matrix,
                param=feat,
                model=FLAGS.flame_model,
                gender=gender,
                pilot_version=2 if FLAGS.pilot else None,
            )

    # Compute similarity matrices based on VGG-Face features
    for data_mode in ["2d-original", "3d-reconstructions"]:  # add VGG feature maps
        data_mode_suffix = "org" if "orig" in data_mode else "3D-recon" if "3d-recon" in data_mode else "3D-persp"
        for layer_name in get_vgg_face_model(save_layer_output=False).layer_names:
            vgg_feat_name = f"VGG_{data_mode_suffix}_{layer_name.upper()}"
            feature_dict[vgg_feat_name] = partial(
                compute_vgg_feature_map_similarity_matrix,
                layer_name=layer_name,
                data_mode=data_mode,
                pilot_version=2 if FLAGS.pilot else None,
            )
            # Compute for within gender stimuli
            for gender in params.GENDERS:
                feature_dict[f"{vgg_feat_name}_{gender}_only"] = partial(
                    compute_vgg_feature_map_similarity_matrix,
                    layer_name=layer_name,
                    data_mode=data_mode,
                    gender=gender,
                    pilot_version=2 if FLAGS.pilot else None,
                )

    # Compute similarity matrices based on SPoSe & VICE (i.e., sparse) embeddings
    for session in params.SESSIONS:
        for sparse_model_name, compute_sparse_feature_map_similarity_matrix in zip(
            ["SPoSE", "VICE"],
            [compute_spose_feature_map_similarity_matrix, compute_vice_feature_map_similarity_matrix],
            strict=True,
        ):
            # Fill function as for other models compute_*_feature_map_similarity_matrix()
            feature_dict[f"{sparse_model_name}_{session}"] = partial(
                compute_sparse_feature_map_similarity_matrix,
                session=session,
                pilot_version=2 if FLAGS.pilot else None,
            )

            # Compute within exclusive gender stimuli
            for gender in params.GENDERS:
                # Fill function as for other models compute_*_feature_map_similarity_matrix()
                feature_dict[f"{sparse_model_name}_{session}_{gender}_only"] = partial(
                    compute_sparse_feature_map_similarity_matrix,
                    session=session,
                    gender=gender,
                    pilot_version=2 if FLAGS.pilot else None,
                )

    # Compute similarity matrices based on VGGFaceHumanJudgment[FrozenCore] embeddings
    for session in params.SESSIONS:
        model_name_session = get_model_hps(session=session, model_name=None, exclusive_gender_trials=None).model_name
        # 1. Compute similarity based on embeddings (i.e., feature maps of vgg_core_bridge), similar to VGGface above
        feature_dict[f"{model_name_session}_{session}_embedding"] = partial(  # pca & metric fill be passed below
            compute_vgg_human_judgment_feature_map_similarity_matrix,
            session=session,
            model_name=model_name_session,
            data_mode=data_mode,
            gender=None,
            pilot_version=2 if FLAGS.pilot else None,
        )

        # Compute within exclusive gender stimuli
        for gender in params.GENDERS:
            model_name_session_gender = get_model_hps(
                session=session, model_name=None, exclusive_gender_trials=gender
            ).model_name
            feature_dict[f"{model_name_session_gender}_{session}_{gender}_only"] = (
                partial(  # pca & metric fill be passed below
                    compute_vgg_human_judgment_feature_map_similarity_matrix,
                    session=session,
                    model_name=model_name_session_gender,
                    data_mode=data_mode,
                    gender=gender,
                    pilot_version=2 if FLAGS.pilot else None,
                )
            )

        # 2. Compute similarity based on VGGFaceHumanjudgment[FrozenCore] decisions, similar to BSMs
        feature_dict[f"{model_name_session}_{session}_decision"] = (
            compute_similarity_matrix_from_vgg_face_human_judgment_model(
                session=session,
                model_name=model_name_session,
                split_return=False,
                n_faces=None,
                exclusive_gender_trials=None,
                verbose=True,
            )
        )

        # Compute within exclusive gender stimuli
        for gender in params.GENDERS:
            model_name_session_gender = get_model_hps(
                session=session, model_name=None, exclusive_gender_trials=gender
            ).model_name

            feature_dict[f"{model_name_session}_{session}_decision_{gender}_only"] = (
                compute_similarity_matrix_from_vgg_face_human_judgment_model(
                    session=session,
                    model_name=model_name_session_gender,
                    split_return=False,
                    n_faces=params.main.n_faces // 2,
                    exclusive_gender_trials=gender,
                    verbose=True,
                )
            )

    # %% Run through all physical & computational face features, and compute their correlations with BSMs and plot them
    for feature, feat_vals in feature_dict.items():
        cprint(string=f"\n{feature}", col="b", fm="ul")
        gender_feat: bool = "_only" in feature

        # Compute cosine similarity of face features, with and without a PCA version
        if callable(feat_vals):
            feat_sim_mat = feat_vals(pca=False, metric=FLAGS.metric)
            pca_feat_sim_mat = feat_vals(pca=FLAGS.pca_fraction, metric=FLAGS.metric)
            extra_sim_case = False
        else:
            feat_sim_mat = feat_vals
            pca_feat_sim_mat = None
            extra_sim_case = True

        # Plot similarity matrices based on face features
        if FLAGS.plot:
            visualise_matrix(
                face_sim_mat=feat_sim_mat,
                session="rsatoolbox" if FLAGS.rsa_toolbox else "",
                pilot=FLAGS.pilot,
                use_rsatoolbox=FLAGS.rsa_toolbox,
                # vmin=feat_sim_mat.min().round(3), vmax=1,
                fig_name=f"Similarity ({'extra' if extra_sim_case else FLAGS.metric}) of {feature} face features",
                save=FLAGS.save_plots,
            )

            if pca_feat_sim_mat is not None:
                visualise_matrix(
                    face_sim_mat=pca_feat_sim_mat,
                    session="pca_rsatoolbox" if FLAGS.rsa_toolbox else "",
                    pilot=FLAGS.pilot,
                    use_rsatoolbox=FLAGS.rsa_toolbox,  # vmax=pca_feat_sim_mat.max(),
                    fig_name=f"Similarity ({FLAGS.metric}) of {feature} face features ({FLAGS.pca_fraction:.0%}-PCA)",
                    save=FLAGS.save_plots,
                )

        # Plot for each gender (for non-exclusive gender features)
        if FLAGS.plot and not gender_feat:
            for gender, slice_it in zip(
                params.GENDERS, [slice(0, n_female, 1), slice(n_female, None, 1)], strict=True
            ):
                visualise_matrix(
                    face_sim_mat=feat_sim_mat[slice_it, slice_it],
                    session="",
                    pilot=FLAGS.pilot,
                    fig_name=f"{gender} ({'extra' if extra_sim_case else FLAGS.metric}) {feature} feats",
                    use_rsatoolbox=FLAGS.rsa_toolbox,
                    save=FLAGS.save_plots,
                )

                if pca_feat_sim_mat is not None:
                    visualise_matrix(
                        face_sim_mat=pca_feat_sim_mat[slice_it, slice_it],
                        session="",
                        pilot=FLAGS.pilot,
                        fig_name=f"{gender} ({FLAGS.metric}) {feature} feats ({FLAGS.pca_fraction:.0%}-PCA)",
                        use_rsatoolbox=FLAGS.rsa_toolbox,
                        save=FLAGS.save_plots,
                    )

        # Compute correlation of physical or computational face features with behavioral similarity judgments (BSM)
        for session, sim_mat_all in zip(params.SESSIONS, [sim_mat_all_2d, sim_mat_all_3d], strict=True):
            cprint(session, fm="ul")
            if not gender_feat:  # Compute correlation face features with BSM for all trials
                r, p = corr_func(vectorize_similarity_matrix(sim_mat_all), vectorize_similarity_matrix(feat_sim_mat))
                msg = (
                    f"{corr_name} correlation between similarity judgments of {session} & {feature} features: "
                    f"r={r:.3f}, p<={p:.5g}"
                )
                logger.info(msg=msg)

                # Save results to file
                rsa_corr_df.loc[f"{session}_BSM", f"{feature}_r"] = r
                rsa_corr_df.loc[f"{session}_BSM", f"{feature}_p"] = p

                if pca_feat_sim_mat is not None:
                    r, p = corr_func(
                        vectorize_similarity_matrix(sim_mat_all), vectorize_similarity_matrix(pca_feat_sim_mat)
                    )
                    msg = (
                        f"{corr_name} correlation between similarity judgments of {session} & "
                        f"{FLAGS.pca_fraction:.0%}-PCA-{feature} features: r={r:.3f}, p<={p:.5g}"
                    )
                    logger.info(msg=msg)

                    # Save results to file
                    rsa_corr_df.loc[f"{session}_BSM", f"{FLAGS.pca_fraction:.0%}-PCA-{feature}_r"] = r
                    rsa_corr_df.loc[f"{session}_BSM", f"{FLAGS.pca_fraction:.0%}-PCA-{feature}_p"] = p
            else:  # Compute correlation face features with BSM for exclusive gender trials
                for gender in params.GENDERS:
                    if f"_{gender}_" not in feature:
                        continue

                    r, p = corr_func(
                        vectorize_similarity_matrix(sim_mats_by_exclusive_gender_dict[session][gender]),
                        vectorize_similarity_matrix(feat_sim_mat),
                    )
                    msg = (
                        f"{corr_name} correlation between similarity judgments of {session} within {gender} only & "
                        f"{feature} features: r={r:.3f}, p<={p:.5g}"
                    )
                    logger.info(msg=msg)

                    # Save results to file
                    rsa_corr_df.loc[f"{session}_BSM_{gender}_only", f"{feature}_r"] = r
                    rsa_corr_df.loc[f"{session}_BSM_{gender}_only", f"{feature}_p"] = p

                    if pca_feat_sim_mat is not None:
                        r, p = corr_func(
                            vectorize_similarity_matrix(sim_mats_by_exclusive_gender_dict[session][gender]),
                            vectorize_similarity_matrix(pca_feat_sim_mat),
                        )
                        msg = (
                            f"{corr_name} correlation between similarity judgments of {session} within {gender} only "
                            f"& {FLAGS.pca_fraction:.0%}-PCA-{feature} features: r={r:.3f}, p<={p:.5g}"
                        )
                        logger.info(msg=msg)

                        # Save results to file
                        rsa_corr_df.loc[
                            f"{session}_BSM_{gender}_only", f"{FLAGS.pca_fraction:.0%}-PCA-{feature}_r"
                        ] = r
                        rsa_corr_df.loc[
                            f"{session}_BSM_{gender}_only", f"{FLAGS.pca_fraction:.0%}-PCA-{feature}_p"
                        ] = p

            # Correlations within gender (non-exclusive trials)
            for gender, slice_it in zip(
                params.GENDERS, [slice(0, n_female, 1), slice(n_female, None, 1)], strict=True
            ):
                r, p = corr_func(
                    vectorize_similarity_matrix(sim_mat_all[slice_it, slice_it]),
                    vectorize_similarity_matrix(feat_sim_mat if gender_feat else feat_sim_mat[slice_it, slice_it]),
                )
                msg = (
                    f"{corr_name} correlation between similarity judgments of {session} & {feature} features in "
                    f"{gender}s: r={r:.3f}, p<={p:.5g}"
                )
                logger.info(msg=msg)

                # Save results to file
                rsa_corr_df.loc[f"{session}_BSM_{gender}", f"{feature}_r"] = r
                rsa_corr_df.loc[f"{session}_BSM_{gender}", f"{feature}_p"] = p

                if pca_feat_sim_mat is not None:
                    r, p = corr_func(
                        vectorize_similarity_matrix(sim_mat_all[slice_it, slice_it]),
                        vectorize_similarity_matrix(
                            pca_feat_sim_mat if gender_feat else pca_feat_sim_mat[slice_it, slice_it]
                        ),
                    )
                    msg = (
                        f"{corr_name} correlation between similarity judgments of {session} & "
                        f"{FLAGS.pca_fraction:.0%}-PCA-{feature} features in {gender}s: r={r:.3f}, p<={p:.5g}"
                    )
                    logger.info(msg=msg)

                    # Save results to file
                    rsa_corr_df.loc[f"{session}_BSM_{gender}", f"{FLAGS.pca_fraction:.0%}-PCA-{feature}_r"] = r
                    rsa_corr_df.loc[f"{session}_BSM_{gender}", f"{FLAGS.pca_fraction:.0%}-PCA-{feature}_p"] = p

    # Save correlation results to file
    if FLAGS.save_corr:
        p2_corr_df = Path(paths.results.main.rsa, f"{corr_name}_{FLAGS.metric}.csv")
        if p2_corr_df.is_file():
            cprint(string=f"Correlation file '{p2_corr_df}' already exists. It will be overwritten ...", col="y")
        rsa_corr_df.to_csv(p2_corr_df, float_format="%.6g")

    # Plot RSA correlation dataframe
    if FLAGS.plot:
        plot_rsa_corr_df(corr_name=corr_name, metric=FLAGS.metric, corr_df=rsa_corr_df, save=FLAGS.save_plots)
        plot_vgg_correlations(
            corr_name=corr_name, metric=FLAGS.metric, data_mode="3d-reconstructions", max_r=1.0, save=FLAGS.save_plots
        )

    # TODO: consider GLM-based unique variance test  # noqa: FIX002


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run RSA analysis")  # init argument parser

    parser.add_argument(
        "--spearman",
        action=argparse.BooleanOptionalAction,
        default=SPEARMAN,
        help="Use Spearman correlation (default: True), otherwise Pearson.",
    )
    parser.add_argument(
        "-m",
        "--metric",
        default=METRIC,
        type=str,
        help="Define metric for similarity measures ('cosine' OR 'euclidean').",
    )
    parser.add_argument(
        "-f",
        "--pca_fraction",
        default=PCA_FRACTION,
        type=float,
        help="Set fraction of variance which needs to be explained by PCA.",
    )
    parser.add_argument("--flame_model", default=FLAME_MODEL, type=str, help="Set FLAME model ('deca' OR 'flame').")
    parser.add_argument(
        "--save_corr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save correlation table of similarity matrices (default: True).",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot similarity matrices (default: False).",
    )
    parser.add_argument(
        "-r",
        "--rsa_toolbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rsa toolbox for plotting (default: True).",
    )
    parser.add_argument(
        "-s",
        "--save_plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save plots of similarity matrices (default: False).",
    )
    parser.add_argument(
        "-p",
        "--pilot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run analysis on pilot data (default: False).",
    )
    parser.add_argument(
        "-l",
        "--logger_overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing logger file (default: False).",
    )

    parser.add_argument(
        "-n",
        "--notification",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send a notification via mattermost.",
    )

    parser.add_argument(
        "-v", "--verbose", action=argparse.BooleanOptionalAction, default=False, help="Be verbose or not."
    )

    # Parse arguments
    FLAGS, unparsed = parser.parse_known_args()
    # Print FLAGS
    if FLAGS.verbose:
        cprint(string="\nFLAGS:\n", fm="ul")
        pprint(FLAGS.__dict__)
        cprint(string="\nunparsed:\n", fm="ul")
        print(*unparsed, sep="\n")

    # %% Run main
    try:
        main()
    except Exception as e:
        # Send notification
        if FLAGS.notification:
            msg = f"RSA analysis in '{__file__}' failed with error:\n\n{e}"
            response = send_to_mattermost(
                text=msg,
                username=config.PROJECT_NAME,
                incoming_webhook=config.minerva.webhook_in,
                icon_url=config.PROJECT_ICON_URL2,
            )

            if not response.ok:
                raise e
        else:
            raise e

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
