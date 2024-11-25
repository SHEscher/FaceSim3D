# !/usr/bin/env python3
"""
Compute the noise ceiling for human similarity judgments.

In terms of:

    * regression coefficients
    * prediction accuracy

"""

# %% Import
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, pearsonr, spearmanr  # , zscore
from tqdm import tqdm
from ut.ils import cprint

from facesim3d.configs import params, paths
from facesim3d.modeling.compute_similarity import cosine_similarity
from facesim3d.modeling.rsa import (
    SPEARMAN,
    compute_similarity_matrix_from_human_judgments,
    extract_exclusive_gender_trials,  # visualise_matrix
    vectorize_similarity_matrix,
)
from facesim3d.read_data import read_trial_results_of_session, read_trial_results_of_set

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Params
MIN_N_SAMPLES_PER_TRIPLET: int = 5  # minimum number of samples per triplet to calculate max accuracy
CHECK_MORE_SAMPLES: bool = False  # check for more samples per triplet
N_REPS: int = 200  # 1_000  # repetitions for noise ceiling computation
PLOTTING: bool = True  # plot results

# Set correlation function
CORR_FUNC = spearmanr if SPEARMAN else pearsonr
CORR_NAME = CORR_FUNC.__name__[:-1].title()  # "Spearman" OR "Pearson"

# Paths
PATH_TO_ACC_TABLE = Path(paths.results.main.noise_ceiling.accuracy_table)
PATH_TO_R_TABLE = Path(paths.results.main.noise_ceiling.r_table.format(corr_name=CORR_NAME.lower()))


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def check_sampling_mode(sampling_mode: str) -> str:
    """Check the sampling mode."""
    sampling_mode = sampling_mode.lower()
    if sampling_mode not in {"full-sample", "multi-sub-sample"}:
        msg = "Mode must be either 'full-sample' or 'multi-sub-sample'!"
        raise ValueError(msg)
    return sampling_mode


def check_gender_suffix(suffix: str | None) -> str:
    """Check the gender suffix."""
    if suffix:
        suffix = suffix.lower()
        if suffix not in params.GENDERS:
            msg = "suffix must be 'female' OR 'male'!"
            raise ValueError(msg)
        return f"_{suffix}"

    return ""


def get_mea_table():
    """Get the maximal empirical accuracy (`MEA`) table."""
    if PATH_TO_ACC_TABLE.exists():
        mea_df = pd.read_csv(PATH_TO_ACC_TABLE, index_col=["session", "sample_type"])
    else:
        mea_df = pd.DataFrame(columns=["session", "sample_type", "max_acc", "min_n_samples"])
        mea_df = mea_df.set_index(["session", "sample_type"])
    return mea_df


def get_mer_table():
    """Get the maximal empirical R (`MER`) table."""
    if PATH_TO_R_TABLE.exists():
        mer_df = pd.read_csv(PATH_TO_R_TABLE, index_col=["session", "sample_type"])
    else:
        mer_df = pd.DataFrame(
            columns=["session", "sample_type", "min_r", "mean_r", "max_r", "var_r", "max_p_value", "min_n_samples"]
        )
        mer_df = mer_df.set_index(["session", "sample_type"])

    return mer_df


def save_mea_table(mea_df: pd.DataFrame):
    """Save the maximal empirical accuracy (`MEA`) table."""
    # Prepare the path
    PATH_TO_ACC_TABLE.parent.mkdir(parents=True, exist_ok=True)

    # Round values
    mea_df["max_acc"] = mea_df["max_acc"].round(4)

    # Save
    mea_df.to_csv(PATH_TO_ACC_TABLE)


def save_mer_table(mer_df: pd.DataFrame):
    """Save the maximal empirical R (`MER`) table."""
    # Prepare the path
    PATH_TO_R_TABLE.parent.mkdir(parents=True, exist_ok=True)

    # Sort index
    mer_df = mer_df.sort_index()
    print(mer_df[["mean_r"]])  # ['min_r', 'mean_r', 'max_r']

    # Round values
    mer_df[["min_r", "mean_r", "max_r"]] = mer_df[["min_r", "mean_r", "max_r"]].round(4)
    mer_df[["var_r"]] = mer_df[["var_r"]].round(6)
    mer_df[["max_p_value"]] = mer_df[["max_p_value"]].round(8)

    # Save
    cprint(string="\nSaving empirical R df ...", col="b")
    mer_df.to_csv(PATH_TO_R_TABLE)


def compute_maximal_empirical_accuracy_in_triplet(choices: np.ndarray) -> float | np.float64:
    """
    Compute the maximal empirical accuracy (`MEA`) within a resampled triplet.

    Here, `MEA` is defined as the accuracy, when the most frequent choice across all samples of the given
    triplet would always be predicted.

    :param choices: array of choices for the same triplet
    :return: maximal empirical accuracy
    """
    _, ctns = np.unique(choices, return_counts=True)  # _ = vals
    return np.max(ctns) / np.sum(ctns)  # max_accuracy


def compute_maximal_empirical_accuracy_in_session(
    session: str, tr_table_of_session: pd.DataFrame, sampling_mode: str, gender_suffix: str = ""
) -> None:
    """
    Compute the maximal empirical accuracy (`MEA`) for a given session.

    :param session: "2D" or "3D"
    :param tr_table_of_session: trial results table of session
    :param sampling_mode: "full-sample" or "multi-sub-sample"
    :param gender_suffix: gender suffix "female" or "male" (if applicable) else empty string ""
    """
    # Checks
    sampling_mode = check_sampling_mode(sampling_mode=sampling_mode)
    suffix = check_gender_suffix(suffix=gender_suffix)

    # Get MEA table
    mea_df = get_mea_table()

    # Get value counts of triplets in table
    tr_val_ctn_table = (
        tr_table_of_session[["triplet_id", "triplet"]]
        .value_counts()
        .rename_axis(["triplet_id", "triplet"])
        .reset_index(name="counts")
    )

    # Compute max empirical accuracy for each triplet
    # Note for the multi-subsampling set, the triplet_id is different from previous sets
    min_n_samples = MIN_N_SAMPLES_PER_TRIPLET
    while True:
        poss_accs = []  # init
        len_trips = []  # init
        for _triplet, multi_triplet_tr in tr_table_of_session[
            tr_table_of_session.triplet.isin(
                tr_val_ctn_table[  # take triplets sampled multiple times
                    tr_val_ctn_table.counts >= min_n_samples
                ].triplet
            )
        ].groupby("triplet"):
            poss_accs.append(compute_maximal_empirical_accuracy_in_triplet(choices=multi_triplet_tr.head_odd.values))
            len_trips.append(len(multi_triplet_tr.head_odd.values))

        if len(poss_accs) == 0:
            break

        # Weighted aggregation of max empirical accuracy over triplets
        max_acc = np.sum(np.array(poss_accs) * np.array(len_trips)) / np.sum(len_trips)
        cprint(
            string=f"\n{sampling_mode.title()} {session}{suffix}: "
            f"Maximal empirical accuracy over {len(len_trips)} "
            f"triplets (with {min_n_samples} or more samples): {max_acc:.2%}",
            col="g",
        )
        mea_df.loc[(session, f"{sampling_mode}{suffix}"), :] = max_acc, min_n_samples

        if not CHECK_MORE_SAMPLES:
            break
        min_n_samples += 1

    # Save MEA table
    save_mea_table(mea_df=mea_df)


def statistical_difference_maximal_empirical_accuracy_between_sessions(
    tr_table_of_2d: pd.DataFrame,
    tr_table_of_3d: pd.DataFrame,
) -> None:
    """
    Run a significance test of differences between maximal empirical accuracies (MEA) for the two sessions (2D, 3D).

    :param tr_table_of_2d: trial results table of 2D session
    :param tr_table_of_3d: trial results table of 2D session
    """
    poss_accs_dict = {}  # init
    len_trips_dict = {}  # init
    for sess in params.SESSIONS:
        tr_table_of_session = {"2D": tr_table_of_2d, "3D": tr_table_of_3d}[sess]

        # Get value counts of triplets in table
        tr_val_ctn_table = (
            tr_table_of_session[["triplet_id", "triplet"]]
            .value_counts()
            .rename_axis(["triplet_id", "triplet"])
            .reset_index(name="counts")
        )

        # Compute max empirical accuracy for each triplet
        # Note for the multi-subsampling set, the triplet_id is different from previous sets
        poss_accs = []  # init
        len_trips = []  # init
        for _triplet, multi_triplet_tr in tr_table_of_session[
            tr_table_of_session.triplet.isin(
                tr_val_ctn_table[  # take triplets sampled multiple times
                    tr_val_ctn_table.counts >= MIN_N_SAMPLES_PER_TRIPLET
                ].triplet
            )
        ].groupby("triplet"):
            poss_accs.append(compute_maximal_empirical_accuracy_in_triplet(choices=multi_triplet_tr.head_odd.values))
            len_trips.append(len(multi_triplet_tr.head_odd.values))

        # Fill dict
        poss_accs_dict[sess] = poss_accs
        len_trips_dict[sess] = len_trips

    # Run significance test:
    is_normal_dist = True  # init
    for sess in params.SESSIONS:
        # Test for normal distribution
        a_d_test = sm.stats.diagnostic.normal_ad(np.array(poss_accs_dict[sess]))
        if a_d_test[1] < 0.05 / 2:  # Bonferroni-corrected
            cprint(
                string=f"Maximal empirical accuracy (MEA) of {sess} is not normally distributed "
                f"(Anderson-Darling test: {a_d_test[0]:.2f}, p-value={a_d_test[1]:.2g})",
                col="r",
            )
            is_normal_dist = False
    if is_normal_dist:  # not the case
        t_stat, p_val, df = sm.stats.ttest_ind(
            poss_accs_dict["2D"], poss_accs_dict["3D"], alternative="two-sided", usevar="pooled"
        )
        cprint(
            string=f"\nStatistical difference between maximal empirical accuracies (MEA) of 2D & 3D: "
            f"t-statistic(df={df})={t_stat:.2f}, p-value={p_val:.2g}",
            col="g",
        )
    else:
        # Stats comparison for non-normal distributions
        # The Mann-Whitney U test is a non-parametric version of the t-test for independent samples.
        u2d, p_val = mannwhitneyu(poss_accs_dict["2D"], poss_accs_dict["3D"], alternative="two-sided", method="auto")
        nx, ny = len(poss_accs_dict["2D"]), len(poss_accs_dict["3D"])
        assert nx == ny  # noqa: S101
        u3d = nx * ny - u2d

        u_test = np.minimum(u2d, u3d)

        # Calculate effect size
        # 1. Pearson's r (however, not applicable for non-normal distributions)
        se = np.sqrt(nx * ny * (nx + ny + 1) / 12)
        e_u = nx * ny / 2  # expected value
        z = (u_test - e_u) / se  # z-score
        pearson_r = z / np.sqrt(nx + ny)

        # 2. Probability of superiority (better for non-normal distributions)
        ps = u_test / (nx * ny)

        cprint(
            string=f"\nStatistical difference between maximal empirical accuracies (MEA) of 2D & 3D (both N={nx}): "
            f"Mann-Whitney U-Test={u_test:.1f}, Z={z:.2f}, p-value={p_val:.2g}; "
            f"effect size: r={pearson_r:.2f}, ps={ps:.2%}.",
            col="g",
        )


def compute_cross_session_accuracy(
    trial_results_table_2d: pd.DataFrame,
    trial_results_table_3d: pd.DataFrame,
    sampling_mode: str,
    gender_suffix: str = "",
) -> None:
    """
    Compute the cross-session accuracy.

    !!! question
        "Can one predict from trials in the 2D-condition trials in the 3D-condition?"
    In case there are multiple samples of the same tripled ID, the most frequent choice is taken.

    ??? note "Minimal impact of the random choices"
        The partially random choices during the comparison between the two viewing conditions (below) can lead to
        slightly different results in the `'match'` column, when this would be run again.
        Nonetheless, the contribution of these random choices should be small,
        since they should cancel each other out (no-match vs. match) in terms of their impact on the overall accuracy.

    :param trial_results_table_2d: trial results table of the 2D-session
    :param trial_results_table_3d: trial results table of the 3D-session
    :param sampling_mode: "full-sample" or "multi-sub-sample"
    :param gender_suffix: gender suffix "female" or "male" (if applicable) else empty string ""
    """
    # Check triplet IDs
    if set(trial_results_table_2d.triplet) != set(trial_results_table_3d.triplet):
        msg = "Triplets (ID's) must match between tables!"
        raise ValueError(msg)

    if not np.all(
        trial_results_table_2d.sort_values(by=["triplet_id"]).triplet.unique()
        == trial_results_table_3d.sort_values(by=["triplet_id"]).triplet.unique()
    ):
        msg = "Triplet ID to triplet mapping must match between tables!"
        raise ValueError(msg)

    # Check sampling_mode
    sampling_mode = check_sampling_mode(sampling_mode=sampling_mode)
    suffix = check_gender_suffix(suffix=gender_suffix)

    # First create a choice table per session (2D, 3D)
    path_to_choice_table = Path(
        paths.results.main.behavior, f"compare_choices_between_sessions_{sampling_mode}{suffix}.csv"
    )
    if path_to_choice_table.exists():
        choice_table = pd.read_csv(path_to_choice_table, index_col="triplet_id", low_memory=False)

    else:
        choice_table = pd.DataFrame(
            columns=["head_odd_2D", "head_odd_3D", "match"],
            index=np.sort(trial_results_table_2d.triplet_id.unique()),
        )
    choice_table.index.name = "triplet_id"

    # Fill choice table via a majority vote per triplet-ID
    new_entries = False

    for tid in tqdm(
        choice_table.index,
        desc=f"Filling choice table with {sampling_mode}{suffix} data",
        total=len(choice_table),
        colour="#63B456",
    ):
        if not pd.isna(choice_table.loc[tid]).all():
            continue
        new_entries = True

        # Get triplet table for current triplet ID
        tid_tab_2d = trial_results_table_2d.loc[trial_results_table_2d.triplet_id == tid, "head_odd"]
        tid_tab_3d = trial_results_table_3d.loc[trial_results_table_3d.triplet_id == tid, "head_odd"]

        # Get value counts in both conditions
        tid_tab_2d_vc = tid_tab_2d.value_counts()
        tid_tab_3d_vc = tid_tab_3d.value_counts()

        # Keep only max count values
        tid_tab_2d_vc = tid_tab_2d_vc[tid_tab_2d_vc == tid_tab_2d_vc.max()]
        tid_tab_3d_vc = tid_tab_3d_vc[tid_tab_3d_vc == tid_tab_3d_vc.max()]

        if len(tid_tab_2d_vc) > 1 or len(tid_tab_3d_vc) > 1:
            # At least two heads were chosen n times

            # Fill choice table with choices
            choice_table.loc[tid, "head_odd_2D"] = sorted(tid_tab_2d_vc.index.astype(int).tolist())
            choice_table.loc[tid, "head_odd_3D"] = sorted(tid_tab_3d_vc.index.astype(int).tolist())

            if choice_table.loc[tid, "head_odd_2D"] == choice_table.loc[tid, "head_odd_3D"]:
                # Case 1: The most chosen heads are equally distributed across viewing conditions
                choice_table.loc[tid, "match"] = True
                continue

            if len(choice_table.loc[tid, "head_odd_2D"]) == len(choice_table.loc[tid, "head_odd_3D"]):
                # Case 2: In both conditions we have the same number of most chosen heads, but the heads are not equal
                # We randomly draw from the most selected heads & compare them
                choice_table.loc[tid, "match"] = np.random.choice(tid_tab_2d_vc.index) == np.random.choice(
                    tid_tab_3d_vc.index
                )
                continue

            # Case 3: The most chosen heads are not equally distributed across viewing conditions
            # We randomly draw from the most selected heads & compare them
            choice_table.loc[tid, "match"] = np.random.choice(tid_tab_2d_vc.index) == np.random.choice(
                tid_tab_3d_vc.index
            )
        else:
            choice_table.loc[tid, "head_odd_2D"] = int(tid_tab_2d_vc.index[0])
            choice_table.loc[tid, "head_odd_3D"] = int(tid_tab_3d_vc.index[0])
            choice_table.loc[tid, "match"] = (
                choice_table.loc[tid, "head_odd_2D"] == choice_table.loc[tid, "head_odd_3D"]
            )

    # Save choice table if there are new entries
    if new_entries:
        print("There are new entries in the choice_table which will be saved")
        path_to_choice_table.parent.mkdir(parents=True, exist_ok=True)
        choice_table.to_csv(path_to_choice_table)
    else:
        print("There are no new entries in the choice_table.")

    # Get & fill MEA table
    mea_df = get_mea_table()
    mea_df.loc[("both", f"{sampling_mode}{suffix}"), :] = (
        choice_table.match.astype(int).mean(),
        np.minimum(
            trial_results_table_2d.triplet_id.value_counts().min(),
            trial_results_table_3d.triplet_id.value_counts().min(),
        ),
    )

    cprint(
        string=f"\n{sampling_mode.title()}{suffix}: Cross-session accuracy (using the majority vote in "
        f"triplets with multiple samples): {choice_table.match.astype(int).mean():.2%}",
        col="g",
    )

    # Save MEA table
    save_mea_table(mea_df=mea_df)


def get_trial_tables(multi_sub_sample_only: bool = True) -> dict:
    """Get the trial results table for each session."""
    trial_table_dict = {}  # init dict to store trial results tables
    if multi_sub_sample_only:
        for session in params.SESSIONS:
            trial_table_dict[session] = read_trial_results_of_set(
                set_nr=f"{session[0]}.20", clean_trials=True, verbose=False
            )
            # Note that in 2D-table there are more than 5 samples of the following triplet-IDs:
            #
            # > ID:     COUNT
            #   -------------
            #   118:    13
            #   419:    12
            #   803:    12
            #   643:    11
            #   1018:   11
            #   331:    11
            #   62:     11
            #   -------------
            # > SUM:    46
            #
            # This leads to 5,746 trials in total, whereas in the 3D-condition we have the expected 5,700 trials.

    else:
        for session in params.SESSIONS:
            trial_table_dict[session] = read_trial_results_of_session(
                session=session,
                clean_trials=True,
                drop_subsamples=True,  # here we remove the additionally acquired sub-sample from the data
                verbose=False,
            )

    # Proces data
    for session in params.SESSIONS:
        # Select columns
        trial_table_dict[session] = trial_table_dict[session][
            ["triplet_id", "triplet", "head1", "head2", "head3", "head_odd"]
        ]
        # Convert column types
        trial_table_dict[session] = trial_table_dict[session].astype(
            {
                "triplet_id": int,
                "head1": int,
                "head2": int,
                "head3": int,
                "head_odd": int,
            }
        )

    return trial_table_dict


def get_r_stats(ls_r: list) -> tuple:
    """Get min, mean, max, and variance from a list of correlation coefficients."""
    return np.min(ls_r), np.mean(ls_r), np.max(ls_r), np.var(ls_r)


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Compute max empirical accuracy & correlation coefficient for FaceSim data
    if MIN_N_SAMPLES_PER_TRIPLET < 2:  # noqa: PLR2004
        msg = "Minimum number of samples per triplet must be at least 2."
        raise ValueError(msg)

    # Load trial data
    tr_table_dict_sub = get_trial_tables(multi_sub_sample_only=True)
    tr_table_dict_full = get_trial_tables(multi_sub_sample_only=False)

    # %% Compute maximal empirical accuracy (MEA)
    for sess in params.SESSIONS:
        compute_maximal_empirical_accuracy_in_session(
            session=sess,
            tr_table_of_session=tr_table_dict_sub[sess],
            sampling_mode="multi-sub-sample",
            gender_suffix="",
        )

        # Compute this for within gender trials only
        gender_cut = params.main.n_faces // 2  # == 50
        arr_female_only = (tr_table_dict_sub[sess][["head1", "head2", "head3", "head_odd"]] < gender_cut).all(axis=1)
        arr_male_only = (tr_table_dict_sub[sess][["head1", "head2", "head3", "head_odd"]] >= gender_cut).all(axis=1)

        for gender_arr, gender in zip([arr_female_only, arr_male_only], params.GENDERS, strict=True):  # female first
            compute_maximal_empirical_accuracy_in_session(
                session=sess,
                tr_table_of_session=tr_table_dict_sub[sess][gender_arr],
                sampling_mode="multi-sub-sample",
                gender_suffix=gender,
            )

    # Statistical difference between MEA of the two sessions
    statistical_difference_maximal_empirical_accuracy_between_sessions(
        tr_table_of_2d=tr_table_dict_sub["2D"],
        tr_table_of_3d=tr_table_dict_sub["3D"],
    )

    # Compute accuracy, when data of one session is predicted by the data of the other session
    compute_cross_session_accuracy(
        trial_results_table_2d=tr_table_dict_sub["2D"],
        trial_results_table_3d=tr_table_dict_sub["3D"],
        sampling_mode="multi-sub-sample",
        gender_suffix="",
    )

    if ("both", "full-sample") not in get_mea_table().index:  # check, since this takes a while
        compute_cross_session_accuracy(
            trial_results_table_2d=tr_table_dict_full["2D"],
            trial_results_table_3d=tr_table_dict_full["3D"],
            sampling_mode="full-sample",
            gender_suffix="",
        )

    # Compute this for within gender trials only
    for s_mode, tr_dict in zip(
        ["multi-sub-sample", "full-sample"], [tr_table_dict_sub, tr_table_dict_full], strict=True
    ):
        if s_mode == "full-sample" and (
            ("both", f"{s_mode}_male") in get_mea_table().index
            and ("both", f"{s_mode}_female") in get_mea_table().index
        ):
            continue  # compute only once, since this takes a while

        tr_table_2d_male = extract_exclusive_gender_trials(trial_results_table=tr_dict["2D"], gender="male")
        tr_table_2d_female = extract_exclusive_gender_trials(trial_results_table=tr_dict["2D"], gender="female")
        tr_table_3d_male = extract_exclusive_gender_trials(trial_results_table=tr_dict["3D"], gender="male")
        tr_table_3d_female = extract_exclusive_gender_trials(trial_results_table=tr_dict["3D"], gender="female")

        # Female
        compute_cross_session_accuracy(
            trial_results_table_2d=tr_table_2d_female,
            trial_results_table_3d=tr_table_3d_female,
            sampling_mode=s_mode,
            gender_suffix="female",
        )  # 59% in multi-sub-sample

        # Male
        compute_cross_session_accuracy(
            trial_results_table_2d=tr_table_2d_male,
            trial_results_table_3d=tr_table_3d_male,
            sampling_mode=s_mode,
            gender_suffix="male",
        )  # 61% in multi-sub-sample

    # %% Compute max empirical correlation coefficient R (noise ceiling)

    # 0.1) Correlate on full sample between 2D & 3D first
    # To be more accurate, the number of samples in each unique triplet should be the same,
    # otherwise triplets with many samples can skew the similarity values of face-pairs.
    # Therefore, sample `max_n_samples` (see below), and repeat this `N_REPS`-times.

    # R-coefficient table
    expected_r_df = get_mer_table()

    if ("both", "full-sample") not in expected_r_df.index:
        list_of_r: list = []
        list_of_cos: list = []
        p_max: float = 0.0
        max_n_samples = min(
            tr_table_dict_full["2D"].triplet.value_counts()[-1], tr_table_dict_full["3D"].triplet.value_counts()[-1]
        )  # here: 1
        for _ in tqdm(range(N_REPS), desc=f"Compute {CORR_NAME} R between sessions (2D~3D) in full-samples"):
            sim_mat_2d = compute_similarity_matrix_from_human_judgments(
                trial_results_table=tr_table_dict_full["2D"].groupby("triplet").sample(max_n_samples),
                pilot=False,
                split_return=False,
                n_faces=params.main.n_faces,
            )
            sim_mat_3d = compute_similarity_matrix_from_human_judgments(
                trial_results_table=tr_table_dict_full["3D"].groupby("triplet").sample(max_n_samples),
                pilot=False,
                split_return=False,
                n_faces=params.main.n_faces,
            )
            # visualise_matrix(sim_mat_2d, session="2D", use_rsatoolbox=True, save=False)  # noqa: ERA001
            # visualise_matrix(sim_mat_3d, session="3D", use_rsatoolbox=True, save=False)  # noqa: ERA001
            v2d = vectorize_similarity_matrix(sim_mat_2d)
            v3d = vectorize_similarity_matrix(sim_mat_3d)
            r, p = CORR_FUNC(v2d, v3d)
            list_of_r.append(r)
            p_max = max(p_max, p)

            list_of_cos.append(cosine_similarity(v2d, v3d))

        cprint(
            string=f"\nFull-sample: {CORR_NAME} correlation between all similarity judgments of 2D & 3D "
            f"(using max {max_n_samples} samples per triplet): "
            f"R={np.mean(list_of_r):.3f}, p<={p_max:.3g}",
            col="g",
        )
        # print(f"{1 - np.mean(list_of_r) ** 2:.2%} of variance in one session remains unexplained by "
        #       f"the other.")
        # print(f"Full-sample: cosine-similarity between all similarity judgments of 2D & 3D: "
        #       f"cosine-sim={np.mean(list_of_cos):.3f}")

        expected_r_df.loc[("both", "full-sample"), :] = (*get_r_stats(list_of_r), p_max, max_n_samples)

        # Compare to taking all samples per triplet (i.e., without sampling)
        v2d = vectorize_similarity_matrix(
            compute_similarity_matrix_from_human_judgments(
                trial_results_table=tr_table_dict_full["2D"],
                pilot=False,
                split_return=False,
                n_faces=params.main.n_faces,
            )
        )
        v3d = vectorize_similarity_matrix(
            compute_similarity_matrix_from_human_judgments(
                trial_results_table=tr_table_dict_full["3D"],
                pilot=False,
                split_return=False,
                n_faces=params.main.n_faces,
            )
        )
        r, p = CORR_FUNC(v2d, v3d)

        cprint(
            string=f"\nFull-sample: {CORR_NAME} correlation between all similarity judgments of 2D & 3D "
            f"(using all samples per triplet): "
            f"R={np.mean(list_of_r):.3f}, p<={p_max:.3g}",
            col="g",
        )

        expected_r_df.loc[("both", "full-sample_all_triplets_at_once"), :] = (np.nan, r, np.nan, np.nan, p, np.nan)

    # 0.2) Correlate on multi-sub-sample between 2D & 3D
    # Note that this is similar to step 2) (below)
    # Option 0: Take all triplet samples & compute R across sessions, do N_REPS times
    if ("both", "multi-sub-sample") not in expected_r_df.index:
        list_of_r: list = []
        list_of_cos: list = []
        p_max: float = 0.0
        max_n_samples_2d = tr_table_dict_sub["2D"].triplet.value_counts()[-1]  # min number per triplet
        max_n_samples_3d = tr_table_dict_sub["3D"].triplet.value_counts()[-1]  # ... don't sample more
        max_n_samples = min(max_n_samples_2d, max_n_samples_3d)
        if max_n_samples_2d != max_n_samples_3d:
            # Sampling is only necessary if minimum number of samples per triplet differs between sessions
            for _ in tqdm(
                range(N_REPS), desc=f"Compute {CORR_NAME} R between sessions (2D~3D) for multi-sub-sample (Option 0)"
            ):
                v2d = vectorize_similarity_matrix(
                    compute_similarity_matrix_from_human_judgments(
                        trial_results_table=tr_table_dict_sub["2D"].groupby("triplet").sample(max_n_samples_2d),
                        pilot=False,
                        split_return=False,
                        n_faces=params.multisubsample.n_faces,
                    )
                )
                v3d = vectorize_similarity_matrix(
                    compute_similarity_matrix_from_human_judgments(
                        trial_results_table=tr_table_dict_sub["3D"].groupby("triplet").sample(max_n_samples_3d),
                        pilot=False,
                        split_return=False,
                        n_faces=params.multisubsample.n_faces,
                    )
                )
                r, p = CORR_FUNC(v2d, v3d)
                list_of_r.append(r)
                p_max = max(p_max, p)

                cos_sim = cosine_similarity(v2d, v3d)
                # print(f"r={r:.2f}, p<={p:.2g}")  # noqa: ERA001
                # print(f"{cos_sim = :.2f}")  # noqa: ERA001
                list_of_r.append(r)
                list_of_cos.append(cos_sim)

            cprint(
                string=f"\nMulti-sub-sample: {CORR_NAME} correlation between all similarity judgments of "
                f"2D & 3D (using all max n samples per triplet): "
                f"R={np.mean(list_of_r):.3f}, p<={p_max:.3g}",
                col="g",
            )

            # print(f"{1 - np.mean(list_of_r) ** 2:.2%} of variance in one session remains unexplained by "
            #       f"the other.")
            # print(f"Multi-sub-sample: cosine-similarity between all similarity judgments of 2D & 3D: "
            #       f"cosine-sim={np.mean(list_of_cos):.3f}")
            expected_r_df.loc[("both", "multi-sub-sample_opt0"), :] = (*get_r_stats(list_of_r), p_max, max_n_samples)

        # Compare to taking all samples per triplet (i.e., without sampling)
        v2d = vectorize_similarity_matrix(
            compute_similarity_matrix_from_human_judgments(
                trial_results_table=tr_table_dict_sub["2D"],
                pilot=False,
                split_return=False,
                n_faces=params.multisubsample.n_faces,
            )
        )
        v3d = vectorize_similarity_matrix(
            compute_similarity_matrix_from_human_judgments(
                trial_results_table=tr_table_dict_sub["3D"],
                pilot=False,
                split_return=False,
                n_faces=params.multisubsample.n_faces,
            )
        )
        r, p = CORR_FUNC(v2d, v3d)

        cprint(
            string=f"\nMulti-sub-sample: {CORR_NAME} correlation between all similarity judgments of 2D & 3D "
            f"(using all samples per triplet): "
            f"R={np.mean(list_of_r):.3f}, p<={p_max:.3g}",
            col="g",
        )

        expected_r_df.loc[("both", "multi-sub-sample"), :] = (np.nan, r, np.nan, np.nan, p, max_n_samples)

    # 1) Correlate within sessions of multi-sub-sample
    for sess in params.SESSIONS:
        if ((sess, "multi-sub-sample_opt1") not in expected_r_df.index) and (
            (sess, "multi-sub-sample_opt2") not in expected_r_df.index
        ):
            # Extract session table
            tr_tab_multi_sample = tr_table_dict_sub[sess]
            # tr_val_ctn_table = tr_table_dict_sub[sess][  # noqa: ERA001, RUF100
            #     ["triplet_id", "triplet"]].value_counts().rename_axis(
            #     ["triplet_id", "triplet"]).reset_index(name='counts')
            # tr_tab_multi_sample = tr_table_dict_sub[sess][tr_table_dict_sub[sess].triplet.isin(  # noqa: E501, ERA0, RUF100
            #     tr_val_ctn_table[tr_val_ctn_table.counts >= MIN_N_SAMPLES_PER_TRIPLET].triplet)]
            # <- not necessary for multi-sub-sample here

            # Option 1: Take from each triplet only 1 & compute R within session, do this N_REPS times
            list_of_r: list = []
            list_of_cos: list = []
            p_max: float = 0.0
            max_n_samples: int = 1
            for _ in tqdm(range(N_REPS), desc=f"Compute noise ceiling for {sess} multi-sub-sample (Option 1)"):
                # Keep from each triplet(_id) only one sample, do this twice
                tr_sample_1 = tr_tab_multi_sample.groupby("triplet").sample(max_n_samples)
                tr_sample_2 = tr_tab_multi_sample.groupby("triplet").sample(max_n_samples)
                # Alternatively, exclude samples from tr_sample_1 in tr_sample_2
                # tr_sample_2 = tr_tab_multi_sample[  # in tr_sample_2 exclude samples from tr_sample_1
                #     ~tr_tab_multi_sample.index.isin(tr_sample_1.index)].groupby("triplet").sample(
                #     max_n_samples)
                # # We don't do this here, because we assume that we sample from the same distribution in both
                # sub-samples. This is not the case if we exclude samples from tr_sample_1 in tr_sample_2.
                # len(tr_sample_1) == len(tr_sample_2)  # noqa: ERA001

                sim_mat_1 = compute_similarity_matrix_from_human_judgments(
                    trial_results_table=tr_sample_1,
                    pilot=False,
                    split_return=False,
                    n_faces=params.multisubsample.n_faces,
                )
                sim_mat_2 = compute_similarity_matrix_from_human_judgments(
                    trial_results_table=tr_sample_2,
                    pilot=False,
                    split_return=False,
                    n_faces=params.multisubsample.n_faces,
                )
                # visualise_matrix(sim_mat_1, session=sess, use_rsatoolbox=True)  # noqa: ERA001
                # visualise_matrix(sim_mat_2, session=sess, use_rsatoolbox=True)  # noqa: ERA001

                # Vectorize similarity matrices
                v1 = vectorize_similarity_matrix(sim_mat_1)
                v2 = vectorize_similarity_matrix(sim_mat_2)

                # Remove NaNs
                if np.all(np.isnan(v1) != np.isnan(v2)):
                    msg = "NaNs in similarity matrices are not in the same locations."
                    raise ValueError(msg)
                v1 = v1[~np.isnan(v1)]
                v2 = v2[~np.isnan(v2)]

                # Compute R and cosine similarity
                r, p = CORR_FUNC(v1, v2)
                p_max = max(p_max, p)
                cos_sim = cosine_similarity(v1, v2)
                list_of_r.append(r)
                list_of_cos.append(cos_sim)

            cprint(string=f"\n{sess} multi-sub-sample {CORR_NAME} R (Option 1):", fm="ul")
            print(pd.Series(list_of_r).describe())
            # cprint(f"\n{sess} multi-sub-sample cosine similarity:", fm="ul")  # noqa: ERA001
            # print(pd.Series(list_of_cos).describe())  # noqa: ERA001

            # Fill in R-table
            expected_r_df.loc[(sess, "multi-sub-sample_opt1"), :] = (*get_r_stats(list_of_r), p_max, np.nan)

            if PLOTTING:
                plt.figure(num=f"{CORR_NAME} R within {sess} multi-sub-sample", figsize=(10, 8))
                _ = plt.hist(list_of_r, bins=len(list_of_r) // 10, label="Opt1", alpha=0.8, color="blue")
                # plt.title(f"Noise ceiling within {sess} multi-sub-sample:\n{CORR_NAME} R of "
                #           f"similarity matrices\nwith {tr_tab_multi_sample.triplet.nunique()} triplets")
                # plt.xlabel(f"{CORR_NAME} R")  # noqa: ERA001
                plt.vlines(
                    x=np.mean(list_of_r),
                    ymin=0,
                    ymax=plt.gca().get_ylim()[1],
                    color="darkblue",
                    alpha=0.5,
                    linestyles=":",
                )

            # Option 2: Split triplets in half, compute R between halves within session, do N_REPS times
            list_of_r: list = []
            list_of_cos: list = []
            p_max: float = 0.0
            max_n_samples: int = tr_tab_multi_sample.triplet.value_counts()[-1] // 2
            for _ in tqdm(range(N_REPS), desc=f"Compute noise ceiling for {sess} multi-sub-sample (Option 2)"):
                # Naive split: take the first half of triplets as one sample, and the second half as another
                # tr_sample_1 = tr_tab_multi_sample.sample(len(tr_tab_multi_sample) // 2, replace=False)  # noqa: E501, ERA001
                # tr_sample_2 = tr_tab_multi_sample[~tr_tab_multi_sample.index.isin(tr_sample_1.index)] # noqa: ERA001

                # Split triplets in half but each triplet should occur in both samples
                tr_sample_1 = tr_tab_multi_sample.groupby("triplet").sample(max_n_samples)
                tr_sample_2 = (
                    tr_tab_multi_sample[  # exclude samples from tr_sample_1
                        ~tr_tab_multi_sample.index.isin(tr_sample_1.index)
                    ]
                    .groupby("triplet")
                    .sample(max_n_samples)
                )
                tr_tab_multi_sample_remain = tr_tab_multi_sample.iloc[  # extract remaining samples
                    list(set(tr_tab_multi_sample.index) - set(tr_sample_1.index) - set(tr_sample_2.index))
                ]

                # Distribute the remaining triplets with multiple samples first
                b = 0  # init
                while tr_tab_multi_sample_remain.triplet.value_counts()[0] > 1:
                    tr = tr_tab_multi_sample_remain.triplet.value_counts().index[0]
                    ctn = tr_tab_multi_sample_remain.triplet.value_counts()[0]

                    if ctn % 2 == 1:
                        b = (ctn % 2 == 1) and not b
                        b_now = int(b)
                    else:
                        b_now = 0

                    tr_sample_1 = tr_sample_1.append(
                        tr_tab_multi_sample_remain[tr_tab_multi_sample_remain.triplet == tr].sample(
                            ctn // 2 + b_now, replace=False
                        )
                    )

                    tr_tab_multi_sample_remain = tr_tab_multi_sample_remain[tr_tab_multi_sample_remain.triplet != tr]

                # Distribute the remaining triplets with one sample now
                tr_sample_1 = tr_sample_1.append(
                    tr_tab_multi_sample_remain.sample(len(tr_tab_multi_sample_remain) // 2, replace=False)
                )

                tr_sample_2 = tr_tab_multi_sample[  # exclude samples from tr_sample_1
                    ~tr_tab_multi_sample.index.isin(tr_sample_1.index)
                ]

                sim_mat_1 = compute_similarity_matrix_from_human_judgments(
                    trial_results_table=tr_sample_1,
                    pilot=False,
                    split_return=False,
                    n_faces=params.multisubsample.n_faces,
                    multi_triplet_mode="ignore",
                    verbose=False,
                )

                sim_mat_2 = compute_similarity_matrix_from_human_judgments(
                    trial_results_table=tr_sample_2,
                    pilot=False,
                    split_return=False,
                    n_faces=params.multisubsample.n_faces,
                    multi_triplet_mode="ignore",
                    verbose=False,
                )

                # visualise_matrix(sim_mat_1, session=sess, use_rsatoolbox=True)  # noqa: ERA001
                # visualise_matrix(sim_mat_2, session=sess, use_rsatoolbox=True)  # noqa: ERA001

                r, p = CORR_FUNC(vectorize_similarity_matrix(sim_mat_1), vectorize_similarity_matrix(sim_mat_2))
                p_max = max(p_max, p)

                cos_sim = cosine_similarity(
                    vectorize_similarity_matrix(sim_mat_1), vectorize_similarity_matrix(sim_mat_2)
                )
                list_of_r.append(r)
                list_of_cos.append(cos_sim)

            cprint(string=f"\n{sess} multi-sub-sample {CORR_NAME} R (Option 2):", fm="ul")
            print(pd.Series(list_of_r).describe())
            # cprint(f"\n{sess} multi-sub-sample cosine similarity:", fm="ul")  # noqa: ERA001
            # print(pd.Series(list_of_cos).describe())  # noqa: ERA001
            # Fill in R-table
            expected_r_df.loc[(sess, "multi-sub-sample_opt2"), :] = (*get_r_stats(list_of_r), p_max, np.nan)

            if PLOTTING:
                # plt.figure(figsize=(10, 8))  # noqa: ERA001
                _ = plt.hist(list_of_r, bins=len(list_of_r) // 10, label="Opt2", color="orange", alpha=0.8)
                plt.vlines(
                    x=np.mean(list_of_r),
                    ymin=0,
                    ymax=plt.gca().get_ylim()[1],
                    color="darkorange",
                    alpha=0.5,
                    linestyles=":",
                )
                plt.title(
                    f"Noise ceiling within {sess} multi-sub-sample:\n{CORR_NAME} R of "
                    f"similarity matrices\nwith {tr_tab_multi_sample.triplet.nunique()} triplets"
                )
                plt.xlabel(f"{CORR_NAME} R")
                plt.legend()
                plt.tight_layout()

    # 2) Compute noise ceiling between sessions (2D, 3D) in multi-sub-sample
    # Option 1: Take from each triplet only 1 sample & compute R across sessions, do this N_REPS times
    if ("both", "multi-sub-sample_opt1") not in expected_r_df.index:
        list_of_r: list = []
        list_of_cos: list = []
        p_max: float = 0.0
        max_n_samples: int = 1
        for _ in tqdm(
            range(N_REPS), desc=f"Compute {CORR_NAME} R between sessions (2D~3D) for multi-sub-sample (Option 1)"
        ):
            # Keep from each triplet(_id) only one sample, do this twice
            tr_sample_2d = tr_table_dict_sub["2D"].groupby("triplet").sample(max_n_samples)
            tr_sample_3d = tr_table_dict_sub["3D"].groupby("triplet").sample(max_n_samples)

            sim_mat_2d = compute_similarity_matrix_from_human_judgments(
                trial_results_table=tr_sample_2d,
                pilot=False,
                split_return=False,
                n_faces=params.multisubsample.n_faces,
            )
            sim_mat_3d = compute_similarity_matrix_from_human_judgments(
                trial_results_table=tr_sample_3d,
                pilot=False,
                split_return=False,
                n_faces=params.multisubsample.n_faces,
            )
            # visualise_matrix(sim_mat_2d, session="2D", use_rsatoolbox=True)  # noqa: ERA001
            # visualise_matrix(sim_mat_3d, session="3D", use_rsatoolbox=True)  # noqa: ERA001

            # Vectorize similarity matrices
            v2d = vectorize_similarity_matrix(sim_mat_2d)
            v3d = vectorize_similarity_matrix(sim_mat_3d)

            # Remove NaNs
            if np.all(np.isnan(v2d) != np.isnan(v3d)):
                msg = "NaNs in similarity matrices are not in the same locations."
                raise ValueError(msg)
            v2d = v2d[~np.isnan(v2d)]
            v3d = v3d[~np.isnan(v3d)]

            # Compute R and cosine similarity
            r, p = CORR_FUNC(v2d, v3d)
            p_max = max(p_max, p)
            cos_sim = cosine_similarity(v2d, v3d)
            list_of_r.append(r)
            list_of_cos.append(cos_sim)

        cprint(string=f"\n2D~3D multi-sub-sample {CORR_NAME} R (Option 1):", fm="ul")
        print(pd.Series(list_of_r).describe())
        # cprint(f"\n{sess} multi-sub-sample cosine similarity:", fm="ul")  # noqa: ERA001
        # print(pd.Series(list_of_cos).describe())  # noqa: ERA001

        # Fill in R-table
        expected_r_df.loc[("both", "multi-sub-sample_opt1"), :] = (*get_r_stats(list_of_r), p_max, max_n_samples)

        if PLOTTING:
            plt.figure(num="R(2D~3D) multi-sub-samples", figsize=(10, 8))
            _ = plt.hist(list_of_r, bins=len(list_of_r) // 10, label="Opt1", alpha=0.8, color="blue")
            # plt.title(f"Noise ceiling within {sess} multi-sub-sample:\n{CORR_NAME} R of "
            #           f"similarity matrices\nwith {tr_tab_multi_sample.triplet.nunique()} triplets")
            # plt.xlabel(f"{CORR_NAME} R")  # noqa: ERA001
            plt.vlines(
                x=np.mean(list_of_r), ymin=0, ymax=plt.gca().get_ylim()[1], color="darkblue", alpha=0.5, linestyles=":"
            )

    # Option 2: Split triplets in half, compute R between halves of both sessions, do N_REPS times
    if ("both", "multi-sub-sample_opt2") not in expected_r_df.index:
        list_of_r: list = []
        list_of_cos: list = []
        p_max: float = 0.0
        n_samples: int = min(len(tr_table_dict_sub["2D"]), len(tr_table_dict_sub["3D"])) // 2
        max_n_samples: int = (
            min(tr_table_dict_sub["2D"].triplet.value_counts()[-1], tr_table_dict_sub["3D"].triplet.value_counts()[-1])
            // 2
        )
        for _ in tqdm(
            range(N_REPS), desc=f"Compute {CORR_NAME} R between sessions (2D~3D) for multi-sub-sample (Option 2)"
        ):
            # Split triplets in half but each triplet should occur in both samples
            tr_sample_2d = tr_table_dict_sub["2D"].groupby("triplet").sample(max_n_samples)
            tr_sample_3d = tr_table_dict_sub["3D"].groupby("triplet").sample(max_n_samples)

            tr_tab_multi_sample_2d_remain = tr_table_dict_sub["2D"].iloc[
                list(set(tr_table_dict_sub["2D"].index) - set(tr_sample_2d.index))
            ]
            tr_tab_multi_sample_3d_remain = tr_table_dict_sub["3D"].iloc[
                list(set(tr_table_dict_sub["3D"].index) - set(tr_sample_3d.index))
            ]

            # Fill with random samples from the remaining triplets for each session
            tr_sample_2d = tr_sample_2d.append(
                tr_tab_multi_sample_2d_remain.sample(n_samples - len(tr_sample_2d), replace=False)
            )
            tr_sample_3d = tr_sample_3d.append(
                tr_tab_multi_sample_3d_remain.sample(n_samples - len(tr_sample_3d), replace=False)
            )

            # Compute similarity matrices
            sim_mat_2d = compute_similarity_matrix_from_human_judgments(
                trial_results_table=tr_sample_2d,
                pilot=False,
                split_return=False,
                n_faces=params.multisubsample.n_faces,
                multi_triplet_mode="ignore",
                verbose=False,
            )
            # params.multisubsample.n_faces == tr_sample_2d.head_odd.nunique()  # noqa: ERA001

            sim_mat_3d = compute_similarity_matrix_from_human_judgments(
                trial_results_table=tr_sample_3d,
                pilot=False,
                split_return=False,
                n_faces=params.multisubsample.n_faces,
                multi_triplet_mode="ignore",
                verbose=False,
            )

            # Compute R and cosine similarity
            r, p = CORR_FUNC(vectorize_similarity_matrix(sim_mat_2d), vectorize_similarity_matrix(sim_mat_3d))
            p_max = max(p_max, p)

            cos_sim = cosine_similarity(
                vectorize_similarity_matrix(sim_mat_2d), vectorize_similarity_matrix(sim_mat_3d)
            )
            list_of_r.append(r)
            list_of_cos.append(cos_sim)

        cprint(string=f"\n2D~3D multi-sub-sample {CORR_NAME} R (Option 2):", fm="ul")
        print(pd.Series(list_of_r).describe())
        # cprint(f"\n2D~3D multi-sub-sample cosine similarity (Option 2):", fm="ul")  # noqa: ERA001
        # print(pd.Series(list_of_cos).describe())  # noqa: ERA001

        # Fill in R-table
        expected_r_df.loc[("both", "multi-sub-sample_opt2"), :] = (*get_r_stats(list_of_r), p_max, max_n_samples)

        if PLOTTING:
            # plt.figure(figsize=(10, 8))  # noqa: ERA001
            _ = plt.hist(list_of_r, bins=len(list_of_r) // 10, label="Opt2", color="orange", alpha=0.8)
            plt.vlines(
                x=np.mean(list_of_r),
                ymin=0,
                ymax=plt.gca().get_ylim()[1],
                color="darkorange",
                alpha=0.5,
                linestyles=":",
            )
            plt.title(f"Noise ceiling for 2D~3D multi-sub-sample:\n{CORR_NAME} R of similarity matrices")
            plt.xlabel(f"{CORR_NAME} R")
            plt.legend()
            plt.tight_layout()

    # 3) Estimate within session noise-ceiling in full sample from multi-sub-samples
    # We take Option 1 (1-sample per triplet for comparability to full-sample)
    mean_r_mss_2d = expected_r_df.loc[("2D", "multi-sub-sample_opt1"), "mean_r"]
    mean_r_mss_3d = expected_r_df.loc[("3D", "multi-sub-sample_opt1"), "mean_r"]
    mean_r_mss_both = expected_r_df.loc[("both", "multi-sub-sample_opt1"), "mean_r"]
    mean_r_fs_both = expected_r_df.loc[("both", "full-sample"), "mean_r"]
    # Compute the ratio of R-values of between-sessions vs. within-sessions
    ratio_mss = np.arctanh(mean_r_mss_both) / np.mean([np.arctanh(mean_r_mss_2d), np.arctanh(mean_r_mss_3d)])
    # Here we apply Fisher transformation using the inverse tangent (arc-tangent) on R-values to make
    # them linearly comparable

    # Compute the estimated noise-ceiling for between-sessions on the full sample
    estimated_mean_r_fs_within = np.tanh(np.arctanh(mean_r_fs_both) / ratio_mss)
    # Use then tangent (tanh) to reverse Fisher transform of this estimate

    cprint(
        string=f"\nEstimated noise-ceiling across sessions 2D~3D in full sample: {estimated_mean_r_fs_within:.3f}",
        col="g",
    )
    # estimated_ratio_fs = estimated_mean_r_fs_within / mean_r_fs_both  ## == ratio_mss  # noqa: ERA001

    # 4) Filter full matrices to only include the same trials as the multi-sub-sampled matrices
    tr_table_dict_full_filtered: dict = {}  # init
    for sess in params.SESSIONS:
        tr_table_dict_full_filtered[sess] = tr_table_dict_full[sess].loc[
            tr_table_dict_full[sess].triplet.isin(tr_table_dict_sub[sess].triplet.unique())
        ]

        if tr_table_dict_full_filtered[sess].triplet.nunique() != np.math.comb(params.multisubsample.n_faces, 3):
            msg = "Here we assume that the full sample contains all possible triplets."
            raise ValueError(msg)

    # tr_table_dict_full_filtered["2D"].triplet.value_counts().hist(alpha=.5)  # noqa: ERA001
    # tr_table_dict_full_filtered["3D"].triplet.value_counts().hist(alpha=.5)  # noqa: ERA001
    if ("both", "downsampled-full-sample_opt1") not in expected_r_df.index:
        list_of_r: list = []
        list_of_cos: list = []
        p_max: float = 0.0
        max_n_samples: int = 1
        for _ in tqdm(
            range(N_REPS),
            desc=f"Compute {CORR_NAME} R between sessions (2D~3D) for downsampled full-sample (Option 1)",
        ):
            v2d = vectorize_similarity_matrix(
                compute_similarity_matrix_from_human_judgments(
                    trial_results_table=tr_table_dict_full_filtered["2D"].groupby("triplet").sample(max_n_samples),
                    pilot=False,
                    split_return=False,
                    n_faces=params.multisubsample.n_faces,
                )
            )

            v3d = vectorize_similarity_matrix(
                compute_similarity_matrix_from_human_judgments(
                    trial_results_table=tr_table_dict_full_filtered["3D"].groupby("triplet").sample(max_n_samples),
                    pilot=False,
                    split_return=False,
                    n_faces=params.multisubsample.n_faces,
                )
            )

            r, p = CORR_FUNC(v2d, v3d)

            # Compute R and cosine similarity
            p_max = max(p_max, p)
            cos_sim = cosine_similarity(v2d, v3d)
            list_of_r.append(r)
            list_of_cos.append(cos_sim)

        expected_r_df.loc[("both", "downsampled-full-sample_opt1"), :] = (
            *get_r_stats(list_of_r),
            p_max,
            max_n_samples,
        )

        cprint(string=f"\n2D~3D downsampled full-sample {CORR_NAME} R (Option 1):", fm="ul")
        print(pd.Series(list_of_r).describe())

    # Per session, sample N_REPS times a subset of the multi-sub-sample tr_table_dict_sub[session]
    #  such that only unique triplets are included.
    #  Then compute R between these submatrices and the filtered full matrices.
    max_n_samples: int = 1
    for sess in params.SESSIONS:
        if (sess, "downsampled-full-and-multi-sub-sample_opt1") not in expected_r_df.index:
            list_of_r: list = []
            p_max: float = 0.0
            for _ in tqdm(
                range(N_REPS),
                desc=f"Compute {CORR_NAME} R between {sess}-session for "
                f"downsampled full-sample & multi-sub-sample (Option 1)",
            ):
                v1 = vectorize_similarity_matrix(
                    compute_similarity_matrix_from_human_judgments(
                        trial_results_table=tr_table_dict_full_filtered[sess].groupby("triplet").sample(max_n_samples),
                        pilot=False,
                        split_return=False,
                        n_faces=params.multisubsample.n_faces,
                    )
                )

                v2 = vectorize_similarity_matrix(
                    compute_similarity_matrix_from_human_judgments(
                        trial_results_table=tr_table_dict_sub[sess].groupby("triplet").sample(max_n_samples),
                        pilot=False,
                        split_return=False,
                        n_faces=params.multisubsample.n_faces,
                    )
                )

                r, p = CORR_FUNC(v1, v2)
                list_of_r.append(r)
                p_max = max(p_max, p)

            print(
                f"Full-sample-reduced ~ Multi-sub-sample {CORR_NAME} correlation "
                f"between all similarity judgments in {sess}: "
            )
            print(f"Mean R of {N_REPS} repetitions (FSS~MSS): {np.mean(list_of_r):.3f}")

            # Fill in table
            expected_r_df.loc[(sess, "downsampled-full-and-multi-sub-sample_opt1"), :] = (
                *get_r_stats(list_of_r),
                p_max,
                max_n_samples,
            )

    # Save R-table for noise ceiling estimates
    save_mer_table(mer_df=expected_r_df)

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
