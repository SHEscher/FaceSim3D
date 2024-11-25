"""Extract feature maps from `VGG-Face` elicited by the original and 3D-reconstructed heads in the `CFD`."""

# %% Import
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ut.ils import get_n_cols_and_rows, oom

from facesim3d.modeling.face_attribute_processing import face_image_path
from facesim3d.modeling.VGG.models import get_vgg_face_model, load_trained_vgg_face_human_judgment_model
from facesim3d.modeling.VGG.prepare_data import load_image_for_model, prepare_data_for_human_judgment_model

if TYPE_CHECKING:
    from facesim3d.modeling.VGG.models import VGGFace, VGGFaceHumanjudgment, VGGFaceHumanjudgmentFrozenCore

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def extract_activation_maps(
    model: VGGFace, image_path: str, layer_name: str | None = None, **kwargs
) -> np.ndarray | list[np.ndarray]:
    """Extract activation map(s) from model layer(s)."""
    model.eval()
    image = load_image_for_model(image_path=image_path, dtype=torch.double, **kwargs)

    # Forward image through VGGFace
    with torch.no_grad():
        _ = model(image)

    if layer_name is None:
        return [l_out.data.cpu().numpy() for l_out in model.layer_output]

    layer_name = layer_name.lower()
    if layer_name not in model.layer_names:
        msg = f"Layer name '{layer_name}' not in model.layer_names !"
        raise ValueError(msg)
    return model.layer_output[model.layer_names.index(layer_name)].data.cpu().numpy()


def get_vgg_activation_maps(list_of_head_nrs: list[str | int], layer_name: str, data_mode: str) -> pd.DataFrame:
    """Get activation maps of the `VGG-Face` model for a list of head models."""
    layer_name = layer_name.lower()
    vgg_model = get_vgg_face_model(save_layer_output=True)
    vgg_model.eval()
    if layer_name not in vgg_model.layer_names:
        msg = f"Layer name '{layer_name}' not in model."
        raise ValueError(msg)

    df_activation_maps = None  # init
    for head_nr in tqdm(
        list_of_head_nrs, desc=f"Extracting activation maps in layer '{layer_name}'", colour="#F79F09"
    ):
        p2_img = face_image_path(head_id=head_nr, data_mode=data_mode, return_head_id=False)
        act_map = extract_activation_maps(model=vgg_model, image_path=p2_img, layer_name=layer_name)
        act_map = act_map.flatten()

        if df_activation_maps is None:
            m_len = len(act_map)
            df_activation_maps = pd.DataFrame(
                index=list_of_head_nrs, columns=[f"{layer_name.upper()}-{i: 0{oom(m_len) + 1}d}" for i in range(m_len)]
            )

        df_activation_maps.loc[head_nr, :] = act_map.astype("float32")

    return df_activation_maps


def extract_vgg_human_judgment_activation_maps_in_core_bridge(
    model: VGGFaceHumanjudgment | VGGFaceHumanjudgmentFrozenCore,
    model_input: torch.Tensor,
) -> np.ndarray | list[np.ndarray]:
    """Extract activation map(s) in the `VGG-core-bridge` from a `VGGFaceHumanjudgment[FrozenCore]` model."""
    model.eval()

    # Forward image through VGGFaceHumanjudgment[FrozenCore]
    with torch.no_grad():
        if model.parallel_bridge:
            # Push the same image through all three parts of the bridge
            out = [
                model.forward_vgg(x=model_input, bridge_idx=1).data.cpu().numpy(),
                model.forward_vgg(x=model_input, bridge_idx=2).data.cpu().numpy(),
                model.forward_vgg(x=model_input, bridge_idx=3).data.cpu().numpy(),
            ]
        else:
            out = model.forward_vgg(x=model_input, bridge_idx=None).data.cpu().numpy()

    return out


def get_vgg_human_judgment_activation_maps(
    list_of_head_nrs: list[str | int] | pd.Series,
    session: str,
    model_name: str,
    data_mode: str,
    exclusive_gender_trials: str | None = None,
) -> pd.DataFrame:
    """Get activation maps of `VGGFaceHumanjudgment[FrozenCore]` for a list of head models."""
    # Get model
    vgg_hj_model = load_trained_vgg_face_human_judgment_model(
        session=session,
        model_name=model_name,
        exclusive_gender_trials=exclusive_gender_trials,
        device="gpu" if torch.cuda.is_available() else "cpu",
    )
    vgg_hj_model.eval()

    # Get model data
    full_dataset_dl, _, _ = prepare_data_for_human_judgment_model(
        session=session,
        frozen_core=vgg_hj_model.freeze_vgg_core,
        data_mode=data_mode,
        last_core_layer=vgg_hj_model.last_core_layer,
        split_ratio=(1.0, 0.0, 0.0),  # push all data in one set
        batch_size=1,
        num_workers=1,
        dtype=torch.float32,
        exclusive_gender_trials=exclusive_gender_trials,
    )

    df_activation_maps = None  # init
    for i, i_data in tqdm(
        enumerate(full_dataset_dl),
        desc="Extracting activation maps in layer 'vgg_core_bridge'",
        total=len(full_dataset_dl),
        colour="#F79F09",
    ):
        ipt_1, ipt_2, ipt_3, _, idx = i_data.values()  # _ == choice
        head_nr_1, head_nr_2, head_nr_3 = (
            f"Head{nr}"
            for nr in full_dataset_dl.dataset.dataset.session_data.iloc[idx.item()][["head1", "head2", "head3"]]
        )

        for head_nr, model_input in zip([head_nr_1, head_nr_2, head_nr_3], [ipt_1, ipt_2, ipt_3], strict=True):
            if head_nr not in (
                list_of_head_nrs.to_numpy() if hasattr(list_of_head_nrs, "to_numpy") else list_of_head_nrs
            ):
                # Skip if the head is not in the list
                continue

            if df_activation_maps is not None and not df_activation_maps.loc[head_nr].isna().any():
                # Skip if this is already computed
                continue

            act_map = extract_vgg_human_judgment_activation_maps_in_core_bridge(
                model=vgg_hj_model, model_input=model_input
            )
            if isinstance(act_map, list):
                act_map = np.concatenate(act_map)
            act_map = act_map.flatten()
            if df_activation_maps is None:
                m_len = len(act_map)
                df_activation_maps = pd.DataFrame(
                    index=list_of_head_nrs, columns=[f"VGG_CORE_BRIDGE-{i: 0{oom(m_len) + 1}d}" for i in range(m_len)]
                )

            df_activation_maps.loc[head_nr, :] = act_map.astype("float32")

    return df_activation_maps


def plot_activation_maps(activation_maps: np.ndarray | list[np.ndarray], layer_names: str | list[str]):
    """Plot activation maps."""
    # Convert to list (if necessary)
    if isinstance(activation_maps, np.ndarray):
        activation_maps = [activation_maps]
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    # Pre-process activation maps for plotting
    processed_layer_outputs = []
    for lout in activation_maps:
        feature_map = lout.squeeze(0)
        gray_scale = np.sum(feature_map, axis=0)
        gray_scale /= feature_map.shape[0]
        processed_layer_outputs.append(gray_scale)

    # Plot activation maps
    n_plots = len([ln for ln in layer_names if "fc" not in ln])
    if n_plots < len(layer_names):
        print(f"Some layers are fully connected (fc) layers (n={len(layer_names) - n_plots}) & will not be plotted.")
    size_r, size_c = get_n_cols_and_rows(n_plots=n_plots, square=True)
    fig = plt.figure(figsize=(9, 9))
    for i, p_lout in enumerate(processed_layer_outputs):
        if "fc" in layer_names[i]:
            continue
        a = fig.add_subplot(size_r, size_c, i + 1)
        _ = plt.imshow(p_lout)
        a.axis("off")
        a.set_title(layer_names[i], fontsize=12)
    plt.tight_layout()

    # TODO: Save figure  # noqa: FIX002
    pass


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
