"""
Extract FLAME parameters from 3D face models.

!!! tip "FLAME model"
    Check out the interactive FLAME model viewer online:

    https://flame.is.tue.mpg.de/interactivemodelviewer.html

"""

# %% Import
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from facesim3d.configs import params, paths
from facesim3d.modeling.face_attribute_processing import head_nr_to_index

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

latent_flame_code = [
    "shape",  # head shape
    "exp",  # expression
    "pose",  # head pose
    "cam",  # camera parameters, since different portraits might have different camera perspectives
    "tex",  # texture
]  # names of the params in the original FLAME implementation
# FLAME model space in DECA: 'Coarse reconstruction'
# DECA specific: "detail" (shape: (1, 128))
#   Individual details (e.g., wrinkles when laughing) in latent code.
#   At a decoding step (not relevant here) this will be used to compute displacements map of individual
#   Ultimately, this comes from a separate ResNet encoder than the other (FLAME) params (for details see
#   Feng et al. 2021)
# Note: There are more, however, multidimensional parameters are not included here.
# Note2: Jozwik et al. (2022) used 3D meshes as well


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def load_deca_params(head_idx_or_nr: str | int):
    """Load estimated FLAME parameters of the DECA model."""
    head_idx = head_nr_to_index(head_idx_or_nr) if str(head_idx_or_nr).startswith("Head") else head_idx_or_nr
    return np.load(
        Path(paths.data.facemodel.deca, f"{head_idx}_inputs_codedict_orig_numpy.npy"), allow_pickle=True
    ).item()


def load_flame_params(head_idx_or_nr: str | int):
    """Load FLAME parameters of the original FLAME implementation."""
    head_idx = head_nr_to_index(head_idx_or_nr) if str(head_idx_or_nr).startswith("Head") else head_idx_or_nr
    return np.load(Path(paths.data.facemodel.flame, f"{head_idx}_inputs_FLAMEfit.npy"), allow_pickle=True).item()


def get_flame_params(list_of_head_nrs: list[str | int], param: str, model: str = "deca") -> pd.DataFrame:
    """Get FLAME parameters (e.g., shape) from the list of head models."""
    model = model.lower()
    assert model in ["deca", "flame"], "Model must be either 'deca' or 'flame'."  # noqa: S101
    assert param in latent_flame_code + (  # noqa: S101
        ["detail"] if model == "deca" else []
    ), f"Parameter must be one of {latent_flame_code}."

    param_loader = load_deca_params if model == "deca" else load_flame_params  # select loader
    param_length = param_loader("Head1")[param].shape[-1]  # get length of parameter from first head
    df_param = pd.DataFrame(
        index=list_of_head_nrs, columns=[f"{param.title()[0]}{i:03d}" for i in range(param_length)]
    )

    # Fill df with FLAME shape params into df
    for head_nr in list_of_head_nrs:
        df_param.loc[head_nr, :] = param_loader(head_nr)[param]

    return df_param


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    flame_params = load_flame_params(1)  # load FLAME params of first head "Head1"
    print("*o" * 5, "FLAME", "*o" * 5)
    for key in flame_params:
        print(f"\t{key}.shape:", flame_params[key].shape)
    print("*o" * 5, "FLAME", "*o" * 5)

    print("\n" + "*o" * 5, "DECA", "*o" * 5)
    deca_params = load_deca_params("Head1")  # the same as 1 (above)
    for key in deca_params:
        print(f"\t{key}.shape:", deca_params[key].shape)
    print("*o" * 5, "DECA", "*o" * 5)

    # TODO: compute difference of FLAME params of DECA and original FLAME implementation  # noqa: FIX002
    pass

    # Explore the DECA shape dimension across all head models
    import matplotlib.pyplot as plt
    from scipy.stats import zscore

    from facesim3d.modeling.prep_computational_choice_model import compute_cosine_similarity_matrix_from_features

    # Collect all shape parameters across all head models
    all_shape_dim = np.empty(shape=(params.main.n_faces, flame_params["shape"].shape[-1]))
    for head_i in range(1, params.main.n_faces + 1):
        deca_params_head_i = load_deca_params(f"Head{head_i}")
        all_shape_dim[head_i - 1] = deca_params_head_i["shape"]

    # Plot all shape parameters
    plt.figure()
    plt.imshow(all_shape_dim)
    plt.xlabel("DECA shape parameters")
    plt.ylabel("Head model")
    plt.title("DECA shape parameters across all head models")

    # Compute similarity matrix of the shape dimension
    deca_shape_sim_mat = compute_cosine_similarity_matrix_from_features(
        features=zscore(all_shape_dim, axis=0)  # z-score each feature column
    )

    # Plot similarity matrix of shape parameters across all head models
    plt.figure()
    plt.imshow(deca_shape_sim_mat)
    plt.title("DECA shape similarity matrix")

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
