"""
Prepare the computational choice model by loading the best weights and computing the similarity matrix.

Author: Simon M. Hofmann
Years: 2023
"""

# %% Import
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ut.ils import browse_files, cprint, find, load_obj, save_obj

from facesim3d.configs import params, paths
from facesim3d.modeling.compute_similarity import compute_cosine_similarity_matrix_from_features

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Hyperparameters (HP) found via HP search
BEST_HP_SPOSE = {"2D": {"lambda": params.spose.lambda_}, "3D": {"lambda": params.spose.lambda_}}
# BEST_HP_VICE is extracted below

PATH_TO_VICE_WEIGHTS = "{modality}/{dims}d/{optim}/{prior}/{spike}/{slab}/{pi}/seed{seed}/"


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def load_spose_weights(
    session: str, gender: str | None = None, pilot: bool = params.PILOT, return_path: bool = False, **hp_kwargs
) -> np.ndarray | tuple[np.ndarray, str]:
    """
    Load weights for the `SPoSE` implementation by the `Vision and Computational Cognition Group` by Martin N. Hebart.

    :param session: '2D', OR '3D'
    :param gender: provide gender ('female', 'male') for model training on exclusive gender trials
    :param pilot: True: use pilot data
    :param return_path: True: return path to the weights-file
    :return: sorted weights
    """
    gender = check_gender_arg(gender=gender)

    # Load sorted weights
    fn = "weights_sorted.npy"
    parent_dir = Path(paths.results.pilot.v2.spose if pilot else paths.results.main.spose, session.upper())

    if len(hp_kwargs) == 0:
        p2_weights = list(parent_dir.glob(f"**/{fn}"))

    else:  # hp is not None:
        # Check if hp is valid
        if "lambda" not in hp_kwargs:
            msg = "Currently only hyperparameter 'lambda' can be processed in load_spose_weights()!"
            raise NotImplementedError(msg)

        p2_weights = list(
            Path(parent_dir).glob(str(Path("".join([f"**/*{value}*/**/" for key, value in hp_kwargs.items()]), fn)))
        )
        if gender is None:
            p2_weights = [p for p in p2_weights if "male/" not in str(p)]
        else:
            p2_weights = [p for p in p2_weights if f"/{gender}" in str(p)]

    if len(p2_weights) == 1:
        p2_weights = p2_weights.pop()

    if isinstance(p2_weights, list):
        cprint(string="Choose one weight file ...", col="b")
        parent_dir = os.path.commonpath(p2_weights)  # find shared parent directory in the list of files
        p2_weights = browse_files(initialdir=parent_dir, filetypes="*.npy")

    # Load weights
    weights = np.load(file=p2_weights)

    if return_path:
        return weights, p2_weights
    return weights


def load_vice_weights(
    session: str,
    pilot: bool = params.PILOT,
    pruned: bool = True,
    return_path: bool = False,
    param_path: str | None = "",
) -> tuple[np.ndarray, np.ndarray, str] | tuple[np.ndarray, np.ndarray]:
    """
    Load parameters for `VICE`.

    !!! note: "How embedding matrices are sorted"
        "pruned parameter matrices are sorted according to their overall importance."

    :param session: '2D', OR '3D'
    :param pilot: True: use pilot data (v2)
    :param pruned: True: return the pruned parameters
    :param return_path: True: return the path to the parameter file
    :param param_path: path to the weights-file, defined by the corresponding VICE params (after /[session]/..)
    :return: parameters (pruned parameters are sorted)
    """
    # Load (sorted) parameters
    parent_dir = Path(paths.results.pilot.v2.vice if pilot else paths.results.main.vice) / session.upper() / param_path
    fn_param = "pruned_params.npz" if pruned else "parameters.npz"

    p2_params = find(fname=fn_param, folder=str(parent_dir), typ="file", abs_path=True, exclusive=False, verbose=False)

    if isinstance(p2_params, list):
        cprint(string=f"Choose a parameter file ('{fn_param}') ...", col="b")
        p2_params = browse_files(initialdir=parent_dir, filetypes="*.npz")

    vice_params = np.load(file=p2_params)

    # Unpack the compressed parameter object
    loc_params = vice_params["pruned_loc" if pruned else "loc"]
    scale_params = vice_params["pruned_scale" if pruned else "scale"]

    if return_path:
        return loc_params, scale_params, p2_params
    return loc_params, scale_params


def check_gender_arg(gender: str | None = None) -> str | None:
    """Check the gender argument."""
    if gender is None:
        return None

    msg = "gender must be 'female' OR 'male' OR None!"
    if isinstance(gender, str):
        gender = gender.lower()
        if gender not in {"female", "male"}:
            raise ValueError(msg)
        return gender

    raise TypeError(msg)


def list_spose_model_performances(
    session: str, gender: str | None = None, modality: str = "behavioral"
) -> pd.DataFrame:
    """
    List `SPoSE` model performances for a given session.

    :param session: '2D', OR '3D'
    :param gender: "female" or "male" OR None
    :param modality: "behavioral" OR others.
    :return: dataframe with the model path, epoch, and validation accuracy sorted by accuracy
    """
    # Check input
    gender = check_gender_arg(gender)

    spose_results_df = pd.DataFrame(columns=["model_path", "epoch", "val_acc"])  # init results df
    sub_folder = modality if gender is None else f"{gender}/{modality}"

    # Find all results.json files
    for results_path in Path(paths.results.main.spose, session, sub_folder).glob("**/results.json"):
        # Get results.json
        with results_path.open() as f:
            spose_result = f.read()

        # Extract accuracy and epoch
        val_acc_at_epoch = json.loads(spose_result)["val_acc"]
        result_epoch = json.loads(spose_result)["epoch"]

        # Add to results dataframe
        new_row = pd.DataFrame(
            [
                {
                    "model_path": str(results_path.parent).split("SPoSE/")[-1],
                    "epoch": result_epoch,
                    "val_acc": val_acc_at_epoch,
                }
            ]
        )
        spose_results_df = pd.concat([spose_results_df, new_row], ignore_index=True)

    # List by accuracy
    return spose_results_df.sort_values(by="val_acc", ascending=False).reset_index(drop=True)


def list_vice_model_performances(
    session: str, gender: str | None = None, modality: str = "behavioral", hp_search: bool = False
) -> pd.DataFrame:
    """
    List `VICE` model performances for a given session.

    :param session: "2D", OR "3D"
    :param gender: "female" or "male" OR None
    :param modality: "behavioral" OR others. Note, hp-search results can be found with "hp_20perc/behavioral".
    :param hp_search: True check the hyperparameter search results
    :return: dataframe with the model path, epoch, and validation accuracy sorted by accuracy
    """
    # Check input
    gender = check_gender_arg(gender)

    vice_results_df = pd.DataFrame(columns=["model_path", "epoch", "val_acc"])  # init results df
    if hp_search and gender is not None:
        cprint(string="gender should be None if searching for HP results.", col="y")
    sub_folder = "hp_20perc/" if hp_search else ""
    sub_folder += modality if gender is None else f"{gender}/{modality}"

    # Find all results*.json files
    for results_path in Path(paths.results.main.vice, session, sub_folder).glob("**/results*.json"):
        # Get results.json
        with results_path.open() as f:
            vice_result = f.read()

        # Extract accuracy and epoch
        val_acc_at_epoch = json.loads(vice_result)["val_acc"]
        result_epoch = json.loads(vice_result)["epoch"]

        # Check if the model is already in the dataframe (since each model has several results_*.json files)
        model_path = str(results_path.parent).split("VICE/")[-1]
        if model_path in vice_results_df["model_path"].values:  # noqa: PD011
            if (
                val_acc_at_epoch
                < vice_results_df.loc[vice_results_df["model_path"] == model_path, "val_acc"].to_numpy()[0]
            ):
                continue  # skip if worse than previous

            # Remove previous entry
            vice_results_df = vice_results_df.drop(vice_results_df[vice_results_df["model_path"] == model_path].index)

        # Add to results dataframe
        vice_results_df = pd.concat(
            [
                vice_results_df,
                pd.DataFrame([{"model_path": model_path, "epoch": result_epoch, "val_acc": val_acc_at_epoch}]),
            ],
            ignore_index=True,
        )

    # List by accuracy
    return vice_results_df.sort_values(by="val_acc", ascending=False).reset_index(drop=True)


def extract_vice_params_from_path(path_to_model_dir: str) -> pd.Series:
    """Extract `VICE` parameters from the path that leads to the model directory."""
    param_keys = ["session", "hp_perc", "modality", "dims", "optim", "prior", "spike", "slab", "pi", "seed"]
    model_params = path_to_model_dir.split("/")
    has_hp_perc = any(("hp_" in pa) for pa in model_params)
    if not has_hp_perc:
        param_keys.remove("hp_perc")

    gender = [g for g in ["female", "male"] if g in model_params]
    if len(gender) > 1:
        msg = "More than one gender found in the path"
        raise ValueError(msg)
    if gender:  # e.g., ["male"]
        import warnings

        warnings.warn(
            message="If gender should be kept, a different implementation is required",
            category=UserWarning,
            stacklevel=2,
        )
        gender = gender.pop()
        cprint(string=f"Remove '{gender}' from model_params!", col="y")
        model_params.remove(gender)

    params_dict = dict(zip(param_keys, model_params, strict=True))

    # Convert params
    if has_hp_perc:
        params_dict["hp_perc"] = int(params_dict["hp_perc"].split("_")[-1].removesuffix("perc"))
    params_dict["dims"] = int(params_dict["dims"].removesuffix("d"))
    params_dict["seed"] = int(params_dict["seed"].removeprefix("seed"))
    for param_key in ["spike", "slab", "pi"]:
        params_dict[param_key] = float(params_dict[param_key])

    return pd.Series(params_dict)


def create_path_from_vice_params(params_dict: dict, gender: str | None = None, pilot: bool = params.PILOT) -> Path:
    """Create a path from `VICE` parameters."""
    gender = check_gender_arg(gender=gender)

    path_from_params = Path(paths.results.pilot.v2.vice if pilot else paths.results.main.vice)

    hp_suffix = ""
    if "hp_perc" in params_dict:
        if gender is not None:
            msg = (
                "There was no gender-exclusive hyperparameter search! "
                "Please set gender to None OR remove 'hp_perc' from params_dict."
            )
            raise ValueError(msg)
        hp_suffix = f"hp_{params_dict['hp_perc']}perc/"

    gender_suffix = ""
    if gender is not None:
        gender_suffix = gender

    return (
        path_from_params
        / params_dict["session"]
        / gender_suffix
        / hp_suffix
        / PATH_TO_VICE_WEIGHTS.format(**params_dict)
    )


def get_best_hp_vice(hp_search: bool = True, print_n: int = 1, from_config: bool = True) -> dict:
    """
    Get the best hyperparameters for `VICE` models.

    :param hp_search: True: get the best hyperparameters from the hyperparameter search
    :param print_n: number of the best hyperparameter sets to print
    :param from_config: True: get the best hyperparameters from the config file
    """
    best_hp_vice = {}  # init
    if from_config and hp_search:
        for sess in params.SESSIONS:
            # They are the same for both conditions (2D, 3D)
            best_hp_vice.update({sess: {}})
            best_hp_vice[sess].update(
                {
                    "session": sess,
                    "hp_perc": params.vice.hp_perc,
                    "modality": params.vice.modality,
                    "dims": params.vice.dims,
                    "optim": params.vice.optim,
                    "prior": params.vice.prior,
                    "spike": params.vice.spike,
                    "slab": params.vice.slab,
                    "pi": params.vice.pi,
                    "seed": params.vice.seed,
                }
            )

        return best_hp_vice

    if from_config and not hp_search:
        cprint(
            string="The config file only contains the best hyperparameters from the hyperparameter search!\n"
            "Will load best hyperparameters from VICE results folder instead ...",
            col="y",
        )

    # Otherwise extract from the VICE results folder
    for session in params.SESSIONS:
        vice_model_table = list_vice_model_performances(session=session, hp_search=hp_search)
        if hp_search:
            vice_model_table = vice_model_table[vice_model_table.model_path.str.contains("hp_20perc")]

        if print_n > 0:
            cprint(string=f"\nBest VICE hyperparameters (descending) | Session: {session}", col="b", fm="ul")
        for _ctn, _p2_model in enumerate(vice_model_table.model_path.values):
            # Extract hyperparameters from the model path
            current_params = extract_vice_params_from_path(_p2_model)

            if print_n > 0:
                print()
                print(current_params.loc[["spike", "slab", "pi"]])
            if _ctn == 0:
                best_hp_vice[current_params.session] = current_params.to_dict()
            if _ctn + 1 == print_n:
                break
    return best_hp_vice


def compute_spose_similarity_matrix(
    session: str, gender: str | None = None, pilot: bool = params.PILOT, save: bool = True
) -> np.ndarray:
    """Compute the similarity matrix of the `SPoSE` model."""
    spose_weights, p2_spose_weights = load_spose_weights(
        session=session, gender=gender, pilot=pilot, return_path=True, **BEST_HP_SPOSE[session]
    )

    spose_sim_mat = compute_cosine_similarity_matrix_from_features(features=spose_weights)

    # Save the SPoSE similarity matrix
    if save:
        save_obj(
            obj=spose_sim_mat,
            name="similarity_matrix",
            folder=p2_spose_weights.parent,
            as_zip=True,
            save_as="npy",
        )

    return spose_sim_mat


def load_best_vice_weights(
    session: str,
    gender: str | None = None,
    hp_search: bool = False,
    pilot: bool = params.PILOT,
    pruned: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, str]:
    """Load the best `VICE` weights."""
    # Get the best hyperparameters for VICE
    best_hp_vice = get_best_hp_vice(hp_search=True, print_n=int(verbose))[session]
    # take the best from the hp search

    if not hp_search:
        # We have to attach the hp_search percentage to the path if we want to load these weights
        best_hp_vice.pop("hp_perc")

    path_to_vice_sim_mat = create_path_from_vice_params(params_dict=best_hp_vice, gender=gender, pilot=pilot)
    # Cut Path at VICE/
    param_path_vice = str(path_to_vice_sim_mat).split(f"VICE/{session}/")[-1]

    vice_weights, _, p2_vice_weights = load_vice_weights(
        session=session,
        pilot=pilot,
        pruned=pruned,
        return_path=True,
        param_path=param_path_vice,
    )  # take only loc_params

    return vice_weights, p2_vice_weights


def compute_vice_similarity_matrix(
    session: str,
    gender: str | None = None,
    hp_search: bool = False,
    pilot: bool = params.PILOT,
    save: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Compute the similarity matrix of the `VICE` model."""
    # Get the weight matrix / embedding of the best VICE model in the given session
    vice_weights, p2_vice_weights = load_best_vice_weights(
        session=session,
        gender=gender,
        hp_search=hp_search,
        pilot=pilot,
        pruned=True,
        verbose=verbose,
    )

    vice_sim_mat = compute_cosine_similarity_matrix_from_features(features=vice_weights)

    # Save the VICE similarity matrix
    if save:
        save_obj(
            obj=vice_sim_mat,
            name="similarity_matrix",
            folder=Path(p2_vice_weights).parent,
            as_zip=True,
            save_as="npy",
        )

    return vice_sim_mat


def load_vice_similarity_matrix(
    session: str, gender: str | None = None, hp_search: bool = False, pilot: bool = params.PILOT, verbose: bool = False
) -> np.ndarray:
    """
    Load the similarity matrix of the `VICE` model.

    ??? note
        Computing the matrix is rapid. So loading might not be necessary.
    """
    gender = check_gender_arg(gender=gender)

    # Get the best hyperparameters for VICE
    best_hp_vice = get_best_hp_vice(hp_search=True, print_n=int(verbose))[session]  # take the best from the hp search
    if not hp_search:
        # However, we remove the hyperparameter percentage, since we want the results from the main run
        best_hp_vice.pop("hp_perc")

    # Generate the path to similarity matrix from the hyperparameter settings
    path_to_vice_sim_mat = create_path_from_vice_params(params_dict=best_hp_vice, gender=gender, pilot=pilot)

    try:
        vice_sim_mat = load_obj(name="similarity_matrix", folder=path_to_vice_sim_mat)
    except FileNotFoundError:
        cprint(string="VICE similarity matrix was not found. Computing it now ...", col="y")
        vice_sim_mat = compute_vice_similarity_matrix(
            session=session,
            gender=gender,
            hp_search=hp_search,
            pilot=pilot,
            save=True,
            verbose=verbose,
        )

    return vice_sim_mat


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
