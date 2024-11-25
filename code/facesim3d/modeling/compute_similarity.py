"""Compute similarity matrices from feature tables."""

# %% Import
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from ut.ils import cprint, normalize

if TYPE_CHECKING:
    import pandas as pd

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def compute_pearson_correlation_between_two_feature_matrices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the Pearson correlation between two feature matrices.

    Both matrices must share at least one dimension with the same length.

    :param x: feature matrix X with shape (N, M)
    :param y: feature matrix Y with shape (N, K)
    :return: correlation matrix of shape (M X K)
    """
    if x.shape[0] != y.shape[0]:
        msg = "Make sure that matrix x comes with shape (N, M), and matrix y with shape (N, K)!"
        raise ValueError(msg)

    # Z-score along common dimension
    x = zscore(x, axis=0)  # shape (N, M)
    y = zscore(y, axis=0)  # shape (N, K)

    # Matrix multiplication along the shared dimension
    r_matrix = x.T @ y  # -> shape (M, K)

    # Divide by product of standard deviation along the shared dimension
    de_x = np.sqrt(np.sum(x**2, axis=0))
    de_y = np.sqrt(np.sum(y**2, axis=0))
    de = de_x.reshape((de_x.shape[0], 1)) @ de_y.reshape((1, de_y.shape[0]))
    # Note, this should be equal to N:
    # np.allclose(a=np.ones((x.shape[1], y.shape[1])) * x.shape[0], b=de)  # noqa: ERA001
    # Note: np.isclose(a=corr_feat_mat[i, j], b=scipy.stats.pearsonr(x=x[:, i], y=y[:, j])[0])  # noqa: ERA001
    return r_matrix / de


def compute_cosine_similarity_matrix_from_features(features: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity matrix according to a given feature matrix.

    :param features: (n_items, m_features)
    :return: cosine similarity matrix (n_items, n_items)
    """
    # Take the magnitude of over model dimensions per face
    magnitude_per_vice_dim = np.linalg.norm(features, axis=1)  # (n_faces, )

    # Outer product of the magnitude
    magnitude_per_cell = np.outer(magnitude_per_vice_dim, magnitude_per_vice_dim)  # (n_faces, n_faces)

    # Dot product of the weights (compare weights/dimensions of each face pair)
    similarity_matrix = features @ features.T  # (n_faces, n_faces)

    # Normalize to get the cosine similarity for each face pair
    return similarity_matrix / magnitude_per_cell  # (n_faces, n_faces)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between two vectors.

    :param vec1: vector 1
    :param vec2: vector 2
    :return: cosine similarity of two vectors
    """
    # np.inner(vec1, vec2) ==  np.dot(vec1, vec2) | vec1 @ vec2.T  # noqa: ERA001
    return np.inner(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance between two vectors.

    :param vec1: vector 1
    :param vec2: vector 2
    :return: Euclidean distance of two vectors
    """
    return np.linalg.norm(vec1 - vec2)


def compute_feature_similarity_matrix(
    feature_table: pd.DataFrame,
    pca: bool | float = False,
    metric: str = "cosine",
    z_score: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Compute the similarity matrix of a given feature table.

    :param feature_table: table with features of heads
    :param pca: False OR provide (0.< pca < 1.) if PCA should be run on feature table with n components such that
                pca [float] *100 % of variance is explained
    :param z_score: True: z-score features before computing similarity matrix
    :param metric: similarity metric to use (cosine, Euclidean)
    :return: similarity matrix
    """
    if pca < 0.0 or pca >= 1.0:
        msg = "pca must be between 0 and 1!"
        raise ValueError(msg)
    metric = metric.lower()
    if metric not in {"cosine", "euclidean"}:
        msg = "metric must be either 'cosine' or 'euclidean'!"
        raise ValueError(msg)

    similarity_metric = cosine_similarity if metric == "cosine" else euclidean_distance

    # Scale features (i.e., z-transform per dimension / column) before computing similarity matrix
    if z_score:
        scaler = StandardScaler().set_output(transform="pandas")
        feature_table = scaler.fit_transform(X=feature_table)
        # this is the same as: scipy.stats.zscore(feature_table.to_numpy(), axis=0)

    # Run PCA (if requested)
    pca_feat_tab = None  # init
    if pca:
        pca_model = PCA(
            n_components=pca,
            svd_solver="full",  # must be 0 < pca < 1 for svd_solver="full" (see docs)
        )  # .set_output(transform="pandas")
        # this finds n components such that pca*100 % of variance is explained
        pca_feat_tab = pca_model.fit_transform(feature_table.to_numpy())  # (n_faces, n_features)
        # pca_model.components_
        explained_variance = pca_model.explained_variance_ratio_.sum()
        print(f"First {pca_model.n_components_} PCA components explain {explained_variance * 100:.1f}% of variance.")
        # f"->\t{pca_model.explained_variance_ratio_}")

    # Compute similarity matrix from given features
    if metric == "cosine":
        try:
            feat_sim_mat = compute_cosine_similarity_matrix_from_features(
                features=feature_table.to_numpy().astype(np.float64) if not pca else pca_feat_tab.astype(np.float64)
            )
            # Take care with interpretation of cosine sim, since 1 == identical, -1 == opposite, 0 == orthogonal.
            # After normalization, this is not the case anymore.
            return normalize(feat_sim_mat, lower_bound=0.0, upper_bound=1.0)
        except TypeError as e:
            cprint(e, col="r")
            pass

    # else use: "euclidean" (or if the cosine computation above failed; this was the case with VGGface features)
    feat_sim_mat = np.identity(len(feature_table))
    for _i, head_pair in enumerate(
        tqdm(
            itertools.combinations(feature_table.index, r=2),
            desc="Computing feature similarity matrix",
            total=np.math.comb(len(feature_table.index), 2),
            colour="#5ED19B",
        )
    ):
        h1, h2 = head_pair
        idx1 = np.where(feature_table.index == h1)[0][0]
        idx2 = np.where(feature_table.index == h2)[0][0]

        # Compute cosine similarity
        if pca:
            similarity = similarity_metric(vec1=pca_feat_tab[idx1], vec2=pca_feat_tab[idx2])
        else:
            similarity = similarity_metric(vec1=feature_table.loc[h1], vec2=feature_table.loc[h2])

        # Fill in matrix
        feat_sim_mat[idx1, idx2] = similarity
        feat_sim_mat[idx2, idx1] = similarity
    # For cosine: feat_sim_mat == compute_cosine_similarity_matrix_from_features(feature_table.to_numpy())

    # Normalize similarity matrix to [0, 1]
    if metric == "euclidean":
        feat_sim_mat[np.diag_indices_from(feat_sim_mat)] = np.nan  # fill diagonal with nan's
        feat_sim_mat = 1 - feat_sim_mat / np.nanmax(feat_sim_mat)  # normalize
        feat_sim_mat[np.diag_indices_from(feat_sim_mat)] = 1.0  # refill diagonal with 1's
    else:
        feat_sim_mat = normalize(feat_sim_mat, lower_bound=0.0, upper_bound=1.0)

    return feat_sim_mat


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
