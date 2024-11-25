# !/usr/bin/env python3
"""
Check data from the triplet-odd-one out task.

This script can be run via the command line interface (CLI).

!!! tip "What arguments can be passed in CLI?"
    ``` bash
    python -m facesim3d.datachecks --help
    ```
"""

# %% Import
from __future__ import annotations

import argparse
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)  # issue with pandas 1.5.1


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from ut.ils import cinput, cprint

from facesim3d.configs import params, paths
from facesim3d.modeling.face_attribute_processing import display_face
from facesim3d.read_data import (
    SET_NUMBER,
    finalized_triplets,
    get_current_state_of_triplets,
    get_list_of_acquired_sets,
    get_triplet_ids_and_heads,
    read_participant_data,
    read_pilot_data,
    read_prolific_participant_data,
    read_trial_results_of_participant,
    read_trial_results_of_session,
    read_trial_results_of_set,
    set_infix,
    update_triplet_table_on_dynamodb,
)

# %% Set global params, & paths < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Set variables
CHECK_BROWSER = False

# Block-lists
block_list_dict = {  # Manually add participants to be blocked (if needed)
    # 2D
    "2.1": [],  # NOTE: DATA REMOVED
    # 3D
    "3.1": [],  # NOTE: DATA REMOVED
}

exception_list_dict = {  # manually add exceptions for participants (if needed)
    # 2D
    "2.1": [],  # NOTE: DATA REMOVED
    # 3D
    "3.1": [],  # NOTE: DATA REMOVED
}

# Default parameters
# (move to `config.toml`, if used in multiple scripts)
BQS_TH = 3  # bad quality score (bqs) threshold (higher than this value is considered bad)
ACCEPT_AT_BQS = 2  # BQS threshold for automatic acceptance of participants
TH_MONO = 6  # threshold for monotonous choices of the same side
TH_MONO_RT = 4.0  # threshold for response time (responses longer than this are not considered as
# monotonous behavior, which is considered as behavior of inattentive participants, who want to save time)
TH_RT = 1.5  # set threshold (in seconds) for rapid response times (all below this is considered rapid)
# could adapt TH_RT based on condition 3D should be a bit longer

W_CATCH = 1.0  # BQS weight for catch trials
W_MONO = 1 / 6  # BQS weight for monotonous choices (currently: 1/TH_MONO)
TH_MONO_FAST_RT = 2.5  # threshold for fast response time during monotonous choice behavior
W_MONO_RT = 1.0  # BQS weight for fast response time during monotonous choice behavior
W_TO = 0.3  # BQS weight for timeouts (i.e., no choice made)
W_RT = 1 / 10  # BQS weight for response time (for each trial with RT <= TH_RT, increment BQS by W_RT)
TH_CONSEC_RT = 3  # threshold for rapid response times that happen consecutively
W_CONSEC_RT = 0.1  # BQS weight for consecutive rapid response times (added on top of RT BQS)


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def check_catch_trials(trial_results_table: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Check catch trials (i.e., those trials, which participants missed)."""
    caught_table = trial_results_table[~trial_results_table.caught.isna()]
    caught_table = caught_table[caught_table.caught == 1]

    if verbose:
        cprint(string="\nCheck catch trials:", fm="ul")
        print(caught_table[["ppid", "trial_num", "head_odd", "triplet", "caught"]])

    return caught_table


def get_choice_side(table_row: pd.Series, verbose: bool = True) -> int | None:
    """Get side of choice in a trial (left, middle, right)."""
    # Could include new keypress column in TrialResults table
    head_cols = ["head1", "head2", "head3", "head_odd"]
    keypress_dict = {"LeftArrow": 0, "DownArrow": 1, "UpArrow": 1, "RightArrow": 2}
    if table_row.keyPress in keypress_dict:
        # keyPress information present
        return keypress_dict[table_row.keyPress]

    if not table_row[head_cols].isna().all():
        side = 0  # init
        max_n_sides = 3
        while side < max_n_sides:
            if table_row[head_cols][side] == table_row[head_cols][-1]:
                return side  # 0: left, 1: middle, 2: right
            side += 1

    if (
        verbose
        and table_row.head_odd != 0
        and not np.isnan(table_row.head_odd)
        and str(int(table_row.head_odd)) not in table_row.triplet
    ):
        # print only if choice was made and triplet does not contain head_odd
        cprint(string=f"Something is wrong in given table row:\n{table_row[head_cols]}", col="r")
        print(table_row)
    return None


def check_monotonous_choices(trial_results_table: pd.DataFrame, set_nr: str, verbose: bool = False) -> pd.DataFrame:
    """Check monotonous choices (i.e., participant chooses repeatedly same side (left, middle, right))."""
    global __monoton_table, __prev_set_nr  # noqa: PLW0603
    # Note: this assumes that each FLAGS.set_nr is run separately (i.e., starting the script freshly)
    try:
        # Check if table is cached as global variable (for multiple calls)
        __monoton_table  # noqa: B018
        if __prev_set_nr != set_nr:
            msg = "Previous set number is different from current set number."
            raise NameError(msg)
    except NameError as e:
        # Iterate through all trials and save indices of monotonous choices (w.r.t. side)
        print(e)
        monotonous_indices = []  # init list of monotonous indices
        cs_cnt = 0  # init choice side (cs) counter
        prev_cs = None  # init previous choice side
        prev_ppid = None  # init previous participant id (ppid)

        for _i, row in tqdm(
            iterable=trial_results_table.iterrows(), desc="Check monotonous choices", total=len(trial_results_table)
        ):
            cs = get_choice_side(table_row=row, verbose=verbose)
            ppid = row.ppid

            if prev_cs is None or cs is None:
                prev_cs = cs
                prev_ppid = ppid
                continue

            # Check whether the previous choice side is equal to current choice side of same participant
            if cs == prev_cs and ppid == prev_ppid and row.response_time <= TH_MONO_RT:
                cs_cnt += 1
            else:
                # Otherwise reset counter
                cs_cnt = 0

            # If there is monotonous choice behavior, add corresponding indices to the list
            if cs_cnt == TH_MONO:
                # Response time could be checked here as well, as an average over the previous trials fast
                # responses with low variance could be a sign of an inattentive participant
                monotonous_indices += list(range(_i + 1 - cs_cnt, _i + 1))
            if cs_cnt > TH_MONO:
                monotonous_indices.append(_i)

            prev_cs = cs
            prev_ppid = ppid

        # Extract monotonous trials
        __monoton_table = trial_results_table.iloc[monotonous_indices]
        __prev_set_nr = set_nr

        if verbose:
            cprint(
                string=f"\nCheck monotonous choices (>={TH_MONO}x same side, response time<={TH_MONO_RT} sec):",
                fm="ul",
            )
            print(__monoton_table[["ppid", "trial_num", "head_odd", "triplet", "response_time"]])

    return __monoton_table


def check_timeouts(trial_results_table: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Check timeouts (i.e., participants do not make any choice in a trial)."""
    rt_table = trial_results_table[trial_results_table.head_odd == 0]

    if verbose:
        cprint(string="\nCheck timeouts:", fm="ul")
        print(rt_table[["ppid", "trial_num", "head_odd", "response_time"]])  # response_time >= 10.

    return rt_table


def check_response_times(trial_results_table: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Check for rapid response times."""
    rt_table = trial_results_table[trial_results_table.response_time <= TH_RT]

    if verbose:
        cprint(string=f"\nCheck response times (<={TH_RT}sec):", fm="ul")
        print(rt_table[["ppid", "trial_num", "response_time"]])

    return rt_table


def get_response_time_stats(trial_results_table: pd.DataFrame, include_pilot: bool = True) -> pd.Series:
    """Get response time statistics."""
    if include_pilot:
        # Aggregate RT stats
        tr_table_pilot = read_pilot_data(clean_trials=True, verbose=False)  # for some comparisons
        return trial_results_table.response_time.append(tr_table_pilot.response_time).describe()

    return trial_results_table.response_time.describe()


def plot_catch_counts(
    trial_results_table: pd.DataFrame, set_nr: str, exclude_training: bool = True, save: bool = False
) -> None:
    """
    Plot catch trial counts.

    :param trial_results_table: Table with trial results from the main study
    :param set_nr: Set number (2D: 2.0, 2.1, ... | 3D: 3.0, 3.1, ...) [str]
    :param exclude_training: Exclude training trials
    :param save: whether to save plot
    :return: None
    """
    plt.figure(
        num=f"Number of missed catch trials in Set{set_nr}" + (" excluding training" if exclude_training else ""),
        figsize=(8, 6),
    )
    catch_table = trial_results_table[trial_results_table.block_num > exclude_training]
    catch_table = catch_table[~catch_table.caught.isna()]  # Remove trials w/o data
    catch_table = catch_table.astype({"caught": bool})
    n_catches = catch_table.groupby("ppid").sum().caught
    if exclude_training:
        n_catches = n_catches.map(lambda x: np.minimum(x, 3))  # cannot be more than 3
    h = sns.histplot(
        x=n_catches,
        discrete=True,
        bins=n_catches.nunique(),
        shrink=0.6,
        color="salmon",
        binrange=(n_catches.min(), n_catches.max()),
    )

    # The following subjects have more than three catch trials missed:
    # catch_table.groupby("ppid").sum()[catch_table.groupby("ppid").sum().caught>3][["correct", "caught"]]  # noqa: E501, ERA001
    # Probably, after 3rd catch and an automatic stop of the experiment, the following catch trials were missed, too,
    # automatically.

    h.get_figure().tight_layout()

    if save:
        h.get_figure().savefig(Path(paths.data.main.qc, f"Set{set_nr}_missed_catch_trials.png"), dpi=300)
        plt.close()
    else:
        plt.show()


def plot_response_time_distribution(
    trial_results_table: pd.DataFrame, set_nr: str, compare_pilot: bool = True
) -> None:
    """
    Plot the response time distribution of a given set.

    :param trial_results_table: table with trial results from the main study
    :param set_nr: Set number (2D: 2.0, 2.1, ... | 3D: 3.0, 3.1, ...) [str]
    :param compare_pilot: Add pilot data in the (aggregated) response time stats, and in the distribution plot
    :return: response time statistics
    """
    # TODO: create session (2D, 3D) specific plots (for both pilot and main data that are compared)  # noqa: FIX002
    #  Session can be inferred from set_nr
    # Get path specifics for given Set
    _, p2_p_table = read_prolific_participant_data(set_nr=set_nr, return_path=True)

    # Plot RT distribution
    p2_rt_dist = Path(
        paths.data.main.qc, f"{p2_p_table.split('/')[-1].split('_')[0]}_Set{set_nr}_response_time_distributions.png"
    )
    if p2_rt_dist.exists():
        # Open image
        Image.open(p2_rt_dist).show()

    else:
        h = None
        # TODO: reduce to participant in given Set  # noqa: FIX002
        plt.figure(num="Response time distributions", figsize=(12, 8))
        for i, (tr_tab, name) in enumerate(
            zip(
                [trial_results_table, read_pilot_data(clean_trials=True)][: compare_pilot + 1],
                ["Main", "Pilot"][: compare_pilot + 1],
                strict=True,
            )
        ):
            # Response time can be max 10 seconds
            tr_tab.response_time = tr_tab.response_time.map(lambda x: np.minimum(x, 10))
            stat = tr_tab.response_time.describe()

            h = sns.kdeplot(
                data=tr_tab,
                x="response_time",
                color=["orange", "blue"][i],
                ax=h,
                label=f"{name} (mean={tr_tab.response_time.mean():.2f}±{tr_tab.response_time.std():.2f}s)",
            )

            for fac in [-1, 0, 1]:  # for 3SD: [-3, -2, -1, 0, 1, 2, 3]
                l_value = stat["mean"] - stat["std"] * fac
                if l_value >= 0:
                    h.axvline(
                        x=l_value,
                        color=["orange", "blue"][i],
                        linestyle="-" if fac == 0 else "--",
                        alpha=0.8 if fac == 0 else 1 / (2 * abs(fac)),
                    )
            print(name, "\n", stat)
        h.set(xlabel="Response time [s]", ylabel="Density", xlim=(0, 11.5))
        h.legend()
        h.get_figure().tight_layout()
        h.get_figure().savefig(p2_rt_dist, dpi=300)


def get_quality_score_table(
    set_nr: str | None = None,
    trial_results_table: pd.DataFrame | None = None,
    save: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute participant bad quality score (BQS).

    This is an aggregate score of the (bad) quality in behavioral responses of single participants.
    """
    # Read the main results table
    save_path = Path(paths.data.main.qc, f"bad_quality_scores_Set{set_nr}.csv")
    if trial_results_table is None:
        trial_results_table = read_trial_results_of_set(set_nr=set_nr, clean_trials=False, verbose=verbose)

        # Check for existing table
        if not save and save_path.exists():  # if save is True, (re-)compute the table
            bqs_tab = pd.read_csv(save_path)
            if set(bqs_tab.ppid) == set(trial_results_table.ppid):
                # Only return table if it contains all participants (otherwise compute scores below)
                if verbose:
                    print("Found existing table.")
                return bqs_tab

    else:
        save = False
        set_nr = f"{np.random.randint(low=4, high=10)}.{np.random.randint(low=4, high=10)}"
        # placeholder set number

    # Get tables of suspicious trials
    c_tab = check_catch_trials(trial_results_table=trial_results_table, verbose=verbose)
    m_tab = check_monotonous_choices(trial_results_table=trial_results_table, set_nr=set_nr, verbose=verbose)
    t_tab = check_timeouts(trial_results_table=trial_results_table, verbose=verbose)
    r_tab = check_response_times(trial_results_table=trial_results_table, verbose=verbose)

    # Compute quality score per participant
    bqs_tab = pd.DataFrame(  # init table w/ aggregate bad quality scores (bqs)
        columns=["ppid", "catch_bqs", "mono_bqs", "timeout_bqs", "response_time_bqs", "BQS"]
    )
    # Add bqs per criterion in table
    for idx, ppid in enumerate(trial_results_table.ppid.unique()):
        bqs_tab.loc[idx, :] = ppid, *[0] * (len(bqs_tab.columns) - 1)  # init participant row

        # Check catch trials
        if c_tab.ppid.str.contains(ppid).any():
            # The higher number of missed catch trials leads to a higher bad quality score (bqs)
            p_c_tab = c_tab.loc[c_tab.ppid == ppid]
            p_c_tab = p_c_tab[p_c_tab.trial_num > params.n_train_trials]  # exclude training trials
            p_bqs = len(p_c_tab)
            bqs_tab.loc[idx, "catch_bqs"] += p_bqs * W_CATCH  # each criterion is weighted by its corresponding weight

        # Check monotonous choices
        if m_tab.ppid.str.contains(ppid).any():
            # The longer the monotonous choice behavior, the higher the bad quality score is
            p_m_tab = m_tab.loc[m_tab.ppid == ppid]
            p_bqs = len(p_m_tab) * W_MONO

            # Add additional bqs for **fast** monotonic responses
            prev_tn = None  # init previous trial number
            sj = p_m_tab.index[0]  # init start index of monotonic choice behavior
            for j, tn in p_m_tab.trial_num.iteritems():
                if tn - 1 != prev_tn and prev_tn is not None:
                    if p_m_tab.loc[sj : j - 1, "response_time"].mean() < TH_MONO_FAST_RT:
                        # Participants choose the same side relatively quickly
                        p_bqs += 1 * W_MONO_RT
                    # this measure could include standard deviation of response times
                    # if p_m_tab.loc[sj:j, "response_time"].std() < .25:  # define threshold
                    #     p_bqs += 1  # noqa: ERA001

                    sj = j  # update start index
                prev_tn = tn  # update previous trial number

            # Add bad quality score to table
            bqs_tab.loc[idx, "mono_bqs"] += p_bqs

        # Check timeouts
        if t_tab.ppid.str.contains(ppid).any():
            # Higher number of timeouts leads to a higher bad quality score (bqs)
            p_t_tab = t_tab.loc[t_tab.ppid == ppid]
            p_t_tab = p_t_tab[p_t_tab.trial_num > params.n_train_trials]  # exclude training trials

            # Check how many timeouts
            p_bqs = len(p_t_tab) * W_TO

            # Could do a measure of whether timeouts happen in consecutive trials?
            pass

            bqs_tab.loc[idx, "timeout_bqs"] += p_bqs

        # Check response times
        if r_tab.ppid.str.contains(ppid).any():
            # High numbers of rapid responses could indicate inattentiveness or sloppy behavior
            p_r_tab = r_tab.loc[r_tab.ppid == ppid]

            # Check how often this happens for this ppid
            p_bqs = len(p_r_tab.response_time) * W_RT

            # Check whether this happens in consecutive trials?
            prev_tn = None  # init previous trial number
            sj = p_r_tab.index[0]  # init start index of rapid choice behavior
            for j, tn in p_r_tab.trial_num.iteritems():
                if tn - 1 != prev_tn and prev_tn is not None:
                    if len(p_r_tab.loc[sj : j - 1]) >= TH_CONSEC_RT:
                        # Participants made consecutively rapid choices
                        p_bqs += len(p_r_tab.loc[sj : j - 1]) * W_CONSEC_RT
                    sj = j  # update start index
                prev_tn = tn  # update previous trial number

            bqs_tab.loc[idx, "response_time_bqs"] += p_bqs

    # Compute aggregate BQS
    bqs_tab.loc[:, "BQS"] = bqs_tab.loc[:, bqs_tab.columns[1:-1]].sum(axis=1)
    for col in bqs_tab.columns:
        if col != "ppid":
            bqs_tab[col] = bqs_tab[col].map("{:,.2f}".format).astype(float)

    if verbose:
        # The lower the quality score is, the better the participant data.
        print("Participants bad quality score (BQS):\n", bqs_tab)

    if save:
        # Save table
        save_path.parent.mkdir(parents=True, exist_ok=True)
        bqs_tab.to_csv(save_path, index=False)

    return bqs_tab


def explore_quality_score_of_single_participant(ppid: str, set_nr: str, trial_results_table: pd.DataFrame) -> None:
    """Explore quality score of single participant."""
    qs_table = get_quality_score_table(set_nr=set_nr, save=False, verbose=False)

    if not qs_table.loc[qs_table.ppid == ppid].empty:
        print("\n" + "-*o*" * 15)
        cprint(string=f"\nQuality scores of ppid '{ppid}':", col="g", fm="ul")
        print(qs_table.loc[qs_table.ppid == ppid, qs_table.columns[1:]])
    else:
        cprint(string=f"Participant '{ppid}' not found in quality score table.", col="r")
        return

    tab = check_catch_trials(trial_results_table=trial_results_table, verbose=False)
    if not tab.loc[tab.ppid == ppid].empty:
        print(
            "\nMissed catch trials:\n",
            tab.loc[tab.ppid == ppid][["trial_num", "response_time", "keyPress", "caught"]],
        )

    tab = check_monotonous_choices(trial_results_table=trial_results_table, set_nr=set_nr, verbose=False)
    if not tab.loc[tab.ppid == ppid].empty:
        print("\nMonotonous choices:\n", tab.loc[tab.ppid == ppid][["trial_num", "keyPress"]])

    tab = check_timeouts(trial_results_table=trial_results_table, verbose=False)
    if not tab.loc[tab.ppid == ppid].empty:
        tab = tab.loc[tab.ppid == ppid].loc[tab.trial_num > params.n_train_trials]
        print("\nTimeouts:\n", tab[["trial_num", "response_time"]])

    tab = check_response_times(trial_results_table=trial_results_table, verbose=False)
    if not tab.loc[tab.ppid == ppid].empty:
        print("\nRapid responses:\n", tab.loc[tab.ppid == ppid][["trial_num", "response_time", "keyPress"]])

    print("\n" + "-*o*" * 15)


def review_prolific_participants(
    set_nr: str,
    ppids_to_review: list[str],
    plot_rt: bool = False,
    accept_at_bqs: float | None = None,
    force_review: bool = False,
) -> None:
    """
    Review participants for low quality data and make a decision for Prolific.

    :param set_nr: Set number (2D: 2.0, 2.1, ... | 3D: 3.0, 3.1, ...) [str]
    :param ppids_to_review: list of Prolific participant IDs to review
    :param plot_rt: whether to plot response time distribution
    :param accept_at_bqs: auto accept participants with BQS below this threshold
    :param force_review: if True: force review of participants even if they have already been reviewed
    :return: table with participants with decision based on BQS and individual factors (behaviour)
    """
    # Load Prolific participant data for given Set
    p_table, p2_p_table = read_prolific_participant_data(set_nr=set_nr, return_path=True)
    ppid_data_table = read_participant_data(process=False)
    trial_results_table = read_trial_results_of_set(set_nr=set_nr, clean_trials=False, verbose=False)
    rt_stat = get_response_time_stats(trial_results_table=trial_results_table, include_pilot=True)
    bqs_tab = get_quality_score_table(set_nr=set_nr, save=False, verbose=False)

    # Define decision categories
    decision_dict = {"r": "reject", "o": "open", "a": "accept", "d": "done"}

    # Attach decision column to Prolific participant table
    d_col = "decision"
    if d_col not in p_table.columns:
        p_table[d_col] = np.nan

    # Review participant by participant
    for ppid in tqdm(ppids_to_review):
        cprint(string=f"\nProcessing participant '{ppid}':", col="b", fm="ul")

        ppid_info = p_table.loc[p_table["Participant id"] == ppid].copy()

        if ppid_info.empty:
            # Participant not in Prolific table, probably due to returned/aborted experiment
            cprint(string=f"No participant information of '{ppid}' in Set{set_nr}.", col="g")
            continue

        current_decision = ppid_info[d_col].item()

        # Check whether participant has to be reviewed
        if not force_review and (
            (current_decision != decision_dict["o"] and not pd.isna(current_decision))
            or ppid_info.Status.item() != "AWAITING REVIEW"
        ):
            # Participant has a decision already
            cprint(string=f"No processing necessary for participant '{ppid}'.", col="g")
            continue
        # else ['REJECTED', 'RETURNED', 'APPROVED']

        # Provide overview of data quality of participant
        explore_quality_score_of_single_participant(ppid=ppid, set_nr=set_nr, trial_results_table=trial_results_table)

        # A comparison the participant's response times to the population mean
        ppid_rt_stat = trial_results_table[trial_results_table.ppid == ppid].response_time.describe()
        print(f"\nParticipant mean RT: {ppid_rt_stat['mean']:.2f} ± {ppid_rt_stat['std']:.2f} sec")
        cprint(string=f"Population  mean RT: {rt_stat['mean']:.2f} ± {rt_stat['std']:.2f} sec", col="g", fm="ul")
        cprint(string=f"Difference  mean RT:{ppid_rt_stat['mean'] - rt_stat['mean']:.2f} sec", col="y", fm="bo")

        # Decide on single participant
        if plot_rt:
            plt.figure(num=f"Response time histogram of '{ppid}'", figsize=(12, 8))
            h = sns.histplot(
                data=trial_results_table[trial_results_table.ppid == ppid],
                x="response_time",
                bins=10 * 10,
                binrange=(0, 10),
            )
            for fac in [-1, 0, 1]:  # for 3SD: [-3, -2, -1, 0, 1, 2, 3]
                l_value = ppid_rt_stat["mean"] - ppid_rt_stat["std"] * fac
                if l_value >= 0:
                    h.axvline(
                        x=l_value, linestyle="-" if fac == 0 else "--", alpha=0.8 if fac == 0 else 1 / (2 * abs(fac))
                    )
            h.set(xlim=(0, 10.3))
            h.get_figure().tight_layout()
            h.get_figure().show()

        # Provide general information about the participant
        if not pd.isna(ppid_info["Time taken"]).values:  # noqa: PD011
            ppid_info["Time taken"] = ppid_info["Time taken"].map(
                lambda x: str(timedelta(seconds=x))
            )  # convert seconds to hh:mm:ss
        print("\n", ppid_info[["Time taken", "Total approvals", "Age", "Sex", "Nationality", "Status"]])

        # Print the self-reported ability to recognize faces if available
        ppid_face_recognition = ppid_data_table[ppid_data_table.ppid == ppid].face_recognition
        if len(ppid_face_recognition) >= 2:  # noqa: PLR2004
            # Find matching Set-Nr in participant data
            infix = f"_{set_infix(set_nr)}_"  # e.g., "_s005_"
            idx = [
                i
                for i, s in ppid_data_table[ppid_data_table.ppid == ppid].iterrows()
                if infix in s.ppid_session_dataname
            ]
            ppid_face_recognition = ppid_face_recognition[idx]

        if len(ppid_face_recognition) != 0 and not ppid_face_recognition.isna().item():
            print(
                f"\nSelf-reported ability to recognise faces [1-7]: \033[95m{ppid_face_recognition.item()}\033[0m"
            )  # print in light magenta

        # Request decision
        if not pd.isna(current_decision):
            cprint(string=f"\nPrevious decision: {current_decision}", col="b")

        # Automatic accept at the given threshold for BQS acceptance
        if accept_at_bqs is not None and bqs_tab.loc[bqs_tab.ppid == ppid].BQS.item() < accept_at_bqs:
            cprint(
                string=f"Automatically accept participant '{ppid}' due to their low "
                f"BQS {bqs_tab.loc[bqs_tab.ppid == ppid].BQS.item():.2f} < {accept_at_bqs:.2f}.",
                col="g",
            )
            decision = "a"
        else:
            try:
                ppid_bqs = bqs_tab.loc[bqs_tab.ppid == ppid].BQS.item()
            except ValueError:
                ppid_bqs = np.nan
            decision = cinput(
                string=f"\nChoose action for participant '{ppid}' "
                f"(BQS={ppid_bqs:.2f}):\n"
                f"\t'[r]eject',\t\t'[o]pen',\t\t'[a]ccept'\n"
                f"Choose__: ",
                col="y",
            ).lower()

        if decision not in decision_dict and decision == "d":
            msg = "Invalid input."
            raise ValueError(msg)

        if plot_rt:
            plt.close()

        # Save decision
        p_table.loc[p_table["Participant id"] == ppid, d_col] = decision_dict[decision]
        p_table.to_csv(p2_p_table, index=False)
        # this overwrites/appends Prolific participant table with decision information


def display_choice_sides(ppid: str, trial_results_table: pd.DataFrame) -> np.ndarray:
    """Display choice sides of a single participant."""
    ppid_tab = trial_results_table.loc[trial_results_table.ppid == ppid]
    ppid_tab = ppid_tab[~ppid_tab.keyPress.isna()].reset_index(drop=True)

    ohe = OneHotEncoder(categories=[["LeftArrow", "DownArrow", "RightArrow"]])
    # Some participants used "UpArrow" instead of "DownArrow"
    ppid_tab.keyPress = ppid_tab.keyPress.replace("UpArrow", "DownArrow")

    ohe.fit(ppid_tab.keyPress.to_numpy().reshape(-1, 1))

    # Transform choices to one-hot encoding
    return ohe.transform(ppid_tab.keyPress.to_numpy().reshape(-1, 1)).toarray()


def prob_catch_trials(r: int, n: int = params.n_catch_trials, verbose: bool = False) -> None:
    """
    Compute the probability of catches equal or lower than r-times (assuming random choice behavior) -> `P(X <= r)`.

    :param r: max number of catches
    :param n: number of catch trials
    :param verbose: add extra explanation
    """
    s = 3  # number of faces in a (catch) trial
    n_o = 1  # number of odd-one-out faces in a catch trial
    p = n_o / s  # probability of choosing the odd-one-out face (i.e., passing the catch trial)
    p_c = (s - n_o) / s  # == 1 - p  # probability of being caught

    # Compute probability of the number of catches being equal or lower than r-times
    px = 0
    for r_i in range(r + 1):
        ncr = np.math.comb(n, r_i)  # n choose r (nCr)  == ncr_p = np.math.comb(n, n-r)
        px += ncr * p_c**r_i * p ** (n - r_i)  # binomial probability P(X=r_i)
    print(f"The chance of having less than {r + 1} catch(es) in {n} catch-trials is: {px:.2%}")

    # Compute probability of number of passed-catch-trials being equal or higher than n-r-times
    if verbose:
        px2 = 0  # px == px2 !
        for r_j in range(n - r, n + 1):
            ncr = np.math.comb(n, r_j)
            px2 += ncr * p**r_j * p_c ** (n - r_j)
        print("\t\t (*** which is equivalent to ***)")
        print(f"The chance of having {n - r} or more pass(es) in {n} catch-trials is: {px2:.2%}")


def display_faces_of_trial(table_row: pd.Series, verbose: bool = False) -> None:
    """Display the faces of a single trial."""
    if table_row[["head1", "head2", "head3"]].isna().any():
        cprint(string=f"Invalid trial of ppid '{table_row.ppid}' (missing face(s)).", col="r")
        return

    faces = [f"Head{int(h)}" for h in table_row[["head1", "head2", "head3"]]]
    choice_side = get_choice_side(table_row=table_row, verbose=verbose)
    catch_head_trial = 0.0
    is_catch_trial = table_row.catch_head != catch_head_trial
    rt = table_row.response_time

    # Display faces
    title = (
        f"PPID '{table_row.ppid}' | Triplet-ID: {int(table_row.triplet_id)} | RT = {rt:.2f} sec | "
        f"trial: {table_row.trial_num}"
    )

    color = "green"
    if is_catch_trial and table_row.caught:
        color = "indianred"  # in case of missed catch trial
        title += " | Caught"

    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, num=title, figsize=(12, 4))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(
            display_face(
                head_id=faces[i], data_mode="3d-reconstructions", interactive=False, show=False, verbose=verbose
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


def display_faces_of_catch_trials(ppid: str, set_nr: str, misses_only: bool = True, verbose: bool = False) -> None:
    """Display the faces of catch trials for given participant."""
    tr_table = read_trial_results_of_participant(ppid=ppid, clean_trials=False, verbose=verbose)
    tr_table = tr_table[~tr_table.caught.isna()]  # remove trials w/o data
    keep_session = f"{ppid}_{set_infix(set_nr)}_trial_results"  # keep data of Set-session only
    tr_table = tr_table.loc[tr_table.ppid_session_dataname == keep_session]

    catch_head_trial = 0.0
    assert (  # noqa: S101
        tr_table.catch_head != catch_head_trial
    ).sum() > 1, "No catch trials found. Do not clean the trial results table before passing it to this function!"

    # Filter for catch trials
    tr_table = tr_table.loc[tr_table.catch_head != catch_head_trial]

    if misses_only:
        tr_table = tr_table.loc[tr_table.caught]

    # Display catch trials
    for _i, table_row in tr_table.iterrows():
        display_faces_of_trial(table_row=table_row, verbose=verbose)


def percentage_missed_n_catch_trials(trial_results_table: pd.DataFrame, n: int = 1) -> None:
    """Compute the percentage of participants who missed at least `n` catch trials."""
    catch_tab = check_catch_trials(trial_results_table=trial_results_table, verbose=False)

    m_ppid_catch = catch_tab[catch_tab.block_num > n].ppid.nunique()
    cprint(
        string=f"{m_ppid_catch / trial_results_table.ppid.nunique():.2%} ({m_ppid_catch} out of "
        f"{trial_results_table.ppid.nunique()}) participants have missed at least {n} catch trial "
        f"in the main phase of the experiment.",
        col="y",
    )


def print_ppid_list_to_accept(set_nr: str) -> None:
    """Print a comma-separated list of ppid's, who are to be accepted on Prolific."""
    prolific_ppid_table = read_prolific_participant_data(set_nr=set_nr)
    if "decision" in prolific_ppid_table.columns:
        ppid_accept_list = prolific_ppid_table.loc[
            (prolific_ppid_table["decision"] == "accept") & (prolific_ppid_table["Status"] == "AWAITING REVIEW")
        ]["Participant id"].to_list()
        cprint(string="\nFollowing participants have been accepted & await review on Prolific:\n", col="g", fm="ul")
        print(*ppid_accept_list, sep=",")  # copy past from console to Prolific

    else:
        cprint(
            string="No 'decision' column in Prolific participant table found. Remaining "
            "{len(prolific_ppid_table.loc[prolific_ppid_table['Status'] == 'AWAITING REVIEW'])} "
            "Participants must be reviewed before!",
            col="r",
        )
    # copy past from console to Prolific


def update_block_list(verbose: bool = False) -> None:
    """Update the blocklist."""
    # Add all blocked participants from all sets
    all_block_ppids = []
    ppids_participated = []
    for set_nr in get_list_of_acquired_sets():
        # Add Set number if missing
        if set_nr not in block_list_dict:
            block_list_dict.update({set_nr: []})
        if set_nr not in exception_list_dict:
            exception_list_dict.update({set_nr: []})

        # Load prolific participant table
        prolific_ppid_table = read_prolific_participant_data(set_nr=set_nr)

        # Add blocked participants
        block_list_dict[set_nr].extend(
            prolific_ppid_table.loc[
                (prolific_ppid_table.Status == "REJECTED") | (prolific_ppid_table.decision == "open")
            ]["Participant id"].to_list()
        )

        # Add high BQS participants (BQS >= 3)
        bqs_table = get_quality_score_table(set_nr=set_nr, save=False)
        block_list_dict[set_nr].extend(bqs_table[BQS_TH <= bqs_table.BQS].ppid.to_list())

        # Add with more than two catches
        tr_table = read_trial_results_of_set(set_nr=set_nr, clean_trials=False, verbose=False)
        tr_table = tr_table[tr_table.block_num > 1]  # remove training trials
        tr_table = tr_table[~tr_table.caught.isna()]  # remove trials w/o data
        ppids_n_catches = tr_table.groupby("ppid").sum().caught
        block_list_dict[set_nr].extend(ppids_n_catches[ppids_n_catches >= 2].index.to_list())  # noqa: PLR2004

        # Add participants which (passed but) have no data
        ppids_wo_data = prolific_ppid_table[~prolific_ppid_table["Participant id"].isin(tr_table.ppid.unique())][
            "Participant id"
        ]
        block_list_dict[set_nr].extend(ppids_wo_data.to_list())

        # Add participants who did the experiment 3 times
        ppids_participated.extend(prolific_ppid_table["Participant id"].to_list())
        ctn_ppids_accepted = pd.value_counts(ppids_participated)
        ppids_with_3_sets = ctn_ppids_accepted[ctn_ppids_accepted >= 3].index.to_list()  # noqa: PLR2004
        block_list_dict[set_nr].extend(ppids_with_3_sets)
        ppids_participated = [ppid for ppid in ppids_participated if ppid not in ppids_with_3_sets]

        # Remove exceptions from blocklist
        for e in exception_list_dict[set_nr]:
            if e in block_list_dict[set_nr]:
                block_list_dict[set_nr].remove(e)

        # Clean up the blocklist: remove duplicates
        block_list_dict[set_nr] = list(set(block_list_dict[set_nr]))  # within sets

        # across sets
        if len(all_block_ppids) == 0:
            all_block_ppids.extend(block_list_dict[set_nr])
        else:
            temp_list = deepcopy(block_list_dict[set_nr])
            duplicates = [ppid for ppid in all_block_ppids if ppid in temp_list]
            for d in duplicates:
                block_list_dict[set_nr].remove(d)
                other_set = next(k for k in block_list_dict if d in block_list_dict[k])
                if verbose:
                    print(f"\t(removing duplicate '{d}' from Set{set_nr}, was already blocked in Set{other_set}.)")

            all_block_ppids.extend(block_list_dict[set_nr])
        if verbose:
            print(f"Set{set_nr} has {len(block_list_dict[set_nr])} blocked participants.")

    if verbose:
        cprint(string=f"Total number of blocked participants: {len(all_block_ppids)}", col="y", fm="bo")


def print_ppid_list_to_block(set_nr: str, per_session: bool = True) -> None:
    """Print a comma separated list of ppid to block on Prolific."""
    # Update list first
    update_block_list(verbose=False)

    # Print all blocklists as one
    session = set_nr[0]  # extract session (2D, 3D)
    str_session = f"{session}D " if per_session else ""
    cprint(
        string=f"\nFollowing participants have been rejected in all acquired {str_session}Sets or "
        f"had no or low quality data & should be excluded in subsequent experiments on "
        f"Prolific:\n",
        col="y",
        fm="ul",
    )
    if per_session:
        print(
            *[
                ppid
                for _set_nr in block_list_dict
                if _set_nr.split(".")[0] == session[0]
                for ppid in block_list_dict[_set_nr]
            ],
            sep=",",
        )
    else:
        print(*[ppid for _set_nr in block_list_dict for ppid in block_list_dict[_set_nr]], sep=",")
    # copy and paste from console to Prolific


def print_ppid_list_for_bonus(trial_results_table: pd.DataFrame, set_nr: str, bonus: float = params.BONUS) -> None:
    """Print a line-separated list of ppid's to provide bonus payments on Prolific."""
    tr_table_wo_training = trial_results_table[trial_results_table.block_num > 1]  # rm training trials
    tr_table_wo_training = tr_table_wo_training[~tr_table_wo_training.caught.isna()]  # Remove trials w/o data
    tr_table_wo_training = tr_table_wo_training.astype({"caught": bool})
    ppids_catches = tr_table_wo_training.groupby("ppid").sum().caught

    # Participants who have no catches (i.e., passed all attention checks) will get a bonus payment
    ppid_bonus_list = ppids_catches[ppids_catches == 0].index.to_list()

    # Append amount of bonus payment to ppid
    payment_for_15_min = 1.94  # (1.94£ == 2.25€)
    bonus_payment = bonus * payment_for_15_min  # 10%
    ppid_bonus_list = [f"{ppid},{bonus_payment:.2f}" for ppid in ppid_bonus_list]

    cprint(
        string=f"Following {len(ppid_bonus_list)} participants in Set{set_nr} will get a bonus payment:\n",
        col="y",
        fm="ul",
    )
    print(*ppid_bonus_list, sep="\n")  # copy past from console to Prolific


def estimate_remaining_costs(in_euro: bool, verbose: bool = False) -> pd.DataFrame:
    """Estimate the remaining costs for the experiment."""
    # Prepare the path for the cost table
    path_to_save = Path(
        paths.data.MAIN,
        "costs",
        f"{datetime.now().date()}_expected_remaining_costs{'_in_euro' if in_euro else ''}.csv",
    )
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    if path_to_save.exists():
        cprint(string=f"Cost table '{path_to_save}' already computed!", col="g")
        return pd.read_csv(path_to_save)

    cprint(string="\nCompute expected remaining costs ...\n", col="b", fm="bo")

    # Provide variables
    # data_loss_per_catch = 1 / 3  # approximately 33%  # noqa: ERA001
    n_trials_per_ppid = 180 - 3 * 3  # 180 trials - (3 blocks * 3 catch trials per block)
    prolific_service_fee = 1 / 3  # 33 %
    cost_per_ppid = 2.25 if in_euro else 1.94  # in € or in £
    perc_bonus_payment = params.BONUS
    non_bonus_sets = [2.0, 2.1, 2.2]

    n_triplets = 0
    n_unseen_triplets = 0
    session_df = pd.DataFrame(
        columns=[
            "n_triplets",
            "n_seen_triplets",
            "n_unseen_triplets",
            "expected_n_trials",
            "data_loss_triplets",
            "ideal_n_ppid",
            "total_expected_n_ppid_triplets",
            "total_expected_n_ppid_trials",
            "n_ppids_approved",
            "data_loss_trials",
            "expected_remaining_n_ppids_triplets",
            "expected_remaining_n_ppids_trials",
            "mean_costs_per_ppid",
            "total_paid",
            "total_expected_costs_triplets",
            "expected_remaining_costs_triplets",
            "total_expected_costs_trials",
            "expected_remaining_costs_trials",
        ],
        index=params.SESSIONS,
    )  # init

    for sess in params.SESSIONS:  # == ["2D", "3D"]
        triplet_tab = get_current_state_of_triplets(session=sess, pilot=False, plot=True)
        sess_n_triplets = len(triplet_tab)  # == np.math.comb(params.main.n_faces, 3) [assert]
        sess_n_unseen_triplets = len(triplet_tab[triplet_tab.status != "G"])
        session_df.loc[sess, ["n_triplets", "n_seen_triplets", "n_unseen_triplets"]] = [
            sess_n_triplets,
            sess_n_triplets - sess_n_unseen_triplets,
            sess_n_unseen_triplets,
        ]
        n_triplets += sess_n_triplets
        n_unseen_triplets += sess_n_unseen_triplets

    # Ideal number of participants that are needed to complete the sampling
    ideal_n_ppid = np.math.ceil(n_triplets / n_trials_per_ppid)
    session_df.loc[:, "ideal_n_ppid"] = session_df.n_triplets.map(lambda x: np.math.ceil(x / n_trials_per_ppid))

    # Estimate cost per participant
    list_of_acquired_sets = get_list_of_acquired_sets()
    n_all_accepted_ppid = 0
    n_ppid_wo_data = 0  # accepted but no data
    df_mean_catch = pd.DataFrame(
        columns=["ppid", "set_nr", "n_blocks", "n_catches", "mean_catch_per_block", "lost_data"]
    )
    df_cost_ppid = pd.DataFrame(columns=["ppid", "set_nr", "cost"])
    for set_nr in list_of_acquired_sets:
        # Extract number of accepted/approved participants
        ppid_table = read_prolific_participant_data(set_nr=set_nr)
        ppid_approved = ppid_table[(ppid_table.Status == "APPROVED") | (ppid_table.decision == "accept")][
            "Participant id"
        ]
        n_all_accepted_ppid += len(ppid_approved)

        # Load trial results table
        tr_table = read_trial_results_of_set(set_nr=set_nr, clean_trials=False, verbose=False)
        tr_table = tr_table[~tr_table.caught.isna()]  # remove trials w/o data
        tr_table = tr_table[tr_table.block_num > 1]  # remove training trials

        # Extract number of participants without data
        ppid_wo_data = ppid_approved[~ppid_approved.isin(tr_table.ppid.unique())]
        n_ppid_wo_data += len(ppid_wo_data)

        # Filter rejected participants
        tr_table = tr_table[tr_table.ppid.isin(ppid_approved)]

        # Compute the average number of catches per block
        for ppid, ppid_tr_table in tr_table.groupby("ppid"):
            n_blocks = ppid_tr_table.block_num.max() - 1
            n_catches = np.minimum(ppid_tr_table.caught.sum(), 3)
            loss = (ppid_tr_table.groupby("block_num").agg({"caught": "sum"}) > 0).mean().values[0]  # noqa: PD011
            # this handles also cases where two catches are in one block and none in the other
            m_catch_per_block = n_catches / n_blocks
            df_mean_catch.loc[len(df_mean_catch), :] = [ppid, set_nr, n_blocks, n_catches, m_catch_per_block, loss]

            ppid_cost = cost_per_ppid
            if set_nr not in non_bonus_sets and n_catches == 0:
                ppid_cost += cost_per_ppid * perc_bonus_payment
            ppid_cost += ppid_cost * prolific_service_fee  # add service fee of Prolific
            df_cost_ppid.loc[len(df_cost_ppid), :] = [ppid, set_nr, round(ppid_cost, 2)]

        # Add data-loss & costs of approved participants w/o trial data
        min_cost_in_set = df_cost_ppid.cost.min()
        for ppid in ppid_wo_data:
            # Add data-loss in catch df
            df_mean_catch.loc[len(df_mean_catch), :] = [ppid, set_nr, np.nan, np.nan, np.nan, 1.0]
            # Add costs
            df_cost_ppid.loc[len(df_cost_ppid), :] = [ppid, set_nr, min_cost_in_set]

        # Compute data-loss based on the expected number of unique triplets in the set
        tr_table = read_trial_results_of_set(set_nr=set_nr, clean_trials=True, verbose=False)
        assert tr_table.equals(  # noqa: S101
            tr_table[tr_table.ppid.isin(ppid_approved)]
        ), f"Trials results are not clean in Set{set_nr}"
        expected_n_triplets_in_set = len(ppid_approved) * n_trials_per_ppid
        n_triplets_in_set = tr_table.triplet_id.nunique()
        data_loss_in_set = 1 - n_triplets_in_set / expected_n_triplets_in_set
        data_loss_in_set_wo_bad_trials = 1 - n_triplets_in_set / len(tr_table)

        if verbose:
            cprint(string=f"\nSet{set_nr}:", fm="ul")
            print(f"Data lost (on trials): {df_mean_catch[df_mean_catch.set_nr == set_nr].lost_data.mean():.1%}")
            print(f"Data lost (on triplets): {data_loss_in_set:.1%}")
            print(f"Data lost (on triplets) w/o bad trials: {data_loss_in_set_wo_bad_trials:.1%}")
            print(f"Costs ({'€' if in_euro else '£'}): {df_cost_ppid[df_cost_ppid.set_nr == set_nr].cost.sum():.2f}")
            print(
                f"Number of approved participants: {len(ppid_approved)} "
                f"({len(ppid_approved) / len(ppid_table):.0%} of ppids on Prolific)"
            )
            print(
                f"Number of approved participants w/o data: {len(ppid_wo_data)} "
                f"({len(ppid_wo_data) / len(ppid_approved):.0%} of approved ppids)"
            )
            print(
                f"Average cost per ppid ({'€' if in_euro else '£'}): "
                f"{df_cost_ppid[df_cost_ppid.set_nr == set_nr].cost.mean():.2f}"
            )

    # Print summary over sessions
    print()
    print("o*+*" * 20)
    for sess in params.SESSIONS:
        print(sess.center(80 // len(params.SESSIONS)), end="")
    print()
    print("o*+*" * 20, "\n")

    cprint(string=f"\nNumber of approved ppids: {n_all_accepted_ppid}", col="g")  # == len(df_cost_ppid)
    df_cost_ppid["session"] = df_cost_ppid.set_nr.apply(lambda x: x.split(".")[0] + "D")
    for sess in params.SESSIONS:
        session_df.loc[sess, "n_ppids_approved"] = len(df_cost_ppid[df_cost_ppid.session == sess])
        cprint(string=f"• {sess}: {int(session_df.loc[sess, 'n_ppids_approved'])}", col="g")

    # Data loss on triplets
    expected_n_trials = n_all_accepted_ppid * n_trials_per_ppid
    n_seen_triplets = n_triplets - n_unseen_triplets
    perc_data_loss_triplets = 1 - (n_seen_triplets / expected_n_trials)
    print(f"Percentage of data loss (on triplets): {perc_data_loss_triplets:.2%}")
    for sess in params.SESSIONS:
        session_df.loc[sess, "expected_n_trials"] = session_df.loc[sess, "n_ppids_approved"] * n_trials_per_ppid
        sess_perc_data_loss_triplets = 1 - (
            session_df.loc[sess, "n_seen_triplets"] / session_df.loc[sess, "expected_n_trials"]
        )
        session_df.loc[sess, "data_loss_triplets"] = sess_perc_data_loss_triplets
        print(f"• in {sess}: {sess_perc_data_loss_triplets:.2%}")
    # perc_data_loss_triplets ≈ session_df.data_loss_triplets.mean()

    # Data loss based on trials
    perc_data_loss_trials = df_mean_catch.lost_data.mean()
    print(f"Percentage of data loss (on trials): {perc_data_loss_trials:.2%}")
    df_mean_catch["session"] = df_mean_catch.set_nr.apply(lambda x: x.split(".")[0] + "D")
    for sess in params.SESSIONS:
        sess_perc_data_loss_trials = df_mean_catch[df_mean_catch.session == sess].lost_data.mean()
        session_df.loc[sess, "data_loss_trials"] = sess_perc_data_loss_trials
        print(f"• in {sess}: {sess_perc_data_loss_trials:.2%}")

    # Compute the required number of participants to finalize sampling based on DynamoDB triplet table
    total_expected_n_ppid_triplets = np.math.ceil(ideal_n_ppid / (1 - perc_data_loss_triplets))
    expected_remaining_n_ppids_triplets = np.math.ceil(
        (n_unseen_triplets / n_trials_per_ppid) / (1 - perc_data_loss_triplets)
    )
    for sess in params.SESSIONS:
        session_df.loc[sess, "total_expected_n_ppid_triplets"] = np.math.ceil(
            session_df.loc[sess, "ideal_n_ppid"] / (1 - session_df.loc[sess, "data_loss_triplets"])
        )
        session_df.loc[sess, "expected_remaining_n_ppids_triplets"] = np.math.ceil(
            (session_df.loc[sess, "n_unseen_triplets"] / n_trials_per_ppid)
            / (1 - session_df.loc[sess, "data_loss_triplets"])
        )

    # Compute the required number of participants to finalize sampling
    total_expected_n_ppid_trials = np.math.ceil(ideal_n_ppid / (1 - perc_data_loss_trials))
    for sess in params.SESSIONS:
        session_df.loc[sess, "total_expected_n_ppid_trials"] = np.math.ceil(
            session_df.loc[sess, "ideal_n_ppid"] / (1 - session_df.loc[sess, "data_loss_trials"])
        )

    print(f"Total number of ppids needed (adjusted for data loss on triplets): {total_expected_n_ppid_triplets}")
    for sess in params.SESSIONS:
        print(f"• {sess}: {int(session_df.loc[sess, 'total_expected_n_ppid_triplets'])}")
    print(f"Total number of ppids needed (adjusted for data loss on trials): {total_expected_n_ppid_trials}")
    for sess in params.SESSIONS:
        print(f"• {sess}: {int(session_df.loc[sess, 'total_expected_n_ppid_trials'])}")

    cprint(
        string=f"Expected remaining number of ppids needed (based on triplets): {expected_remaining_n_ppids_triplets}",
        col="r",
        fm="bo",
    )
    cprint(
        string=f"Expected remaining number of ppids needed (based on trials): "
        f"{total_expected_n_ppid_trials - n_all_accepted_ppid}",
        col="y",
        fm="bo",
    )
    # total_expected_n_ppid_triplets - n_all_accepted_ppid
    for sess in params.SESSIONS:
        cprint(string=f"• {sess}: {int(session_df.loc[sess, 'expected_remaining_n_ppids_triplets'])}", col="r")
        session_df.loc[sess, "expected_remaining_n_ppids_trials"] = (
            session_df.loc[sess, "total_expected_n_ppid_trials"] - session_df.loc[sess, "n_ppids_approved"]
        )
        cprint(string=f"• {sess}: {int(session_df.loc[sess, 'expected_remaining_n_ppids_trials'])}", col="y")

    # Compute costs
    mean_cost_per_ppid = df_cost_ppid.cost.mean()
    print(f"\nAverage cost per ppid ({'€' if in_euro else '£'}): {mean_cost_per_ppid:.2f}")
    for sess in params.SESSIONS:
        session_df.loc[sess, "mean_costs_per_ppid"] = df_cost_ppid[df_cost_ppid.session == sess].cost.mean()
        print(f"• {sess}: {session_df.loc[sess, 'mean_costs_per_ppid']:.2f}")

    if verbose:
        # Percentage of participants that get bonus
        perc_all_pass = len(df_mean_catch[df_mean_catch.n_catches == 0]) / len(df_mean_catch)
        print(f"Participants who get bonus (no-catch): {perc_all_pass:.2%}")
        for sess in params.SESSIONS:
            sess_perc_all_pass = len(
                df_mean_catch[(df_mean_catch.session == sess) & (df_mean_catch.n_catches == 0)]
            ) / len(df_mean_catch[df_mean_catch.session == sess])
            print(f" • {sess}: {sess_perc_all_pass:.2%}")

    paid_costs = df_cost_ppid.cost.sum()
    cprint(string=f"Total paid costs ({'€' if in_euro else '£'}): {paid_costs:.2f}", col="g")
    for sess in params.SESSIONS:
        session_df.loc[sess, "total_paid"] = df_cost_ppid[df_cost_ppid.session == sess].cost.sum()
        # == session_df.n_ppids_approved * session_df.mean_costs_per_ppid
        cprint(string=f"• {sess}: {session_df.loc[sess, 'total_paid']:.2f}", col="g")

    # Compute expected total remaining costs
    total_expected_costs_triplets = mean_cost_per_ppid * total_expected_n_ppid_triplets
    total_expected_costs_trials = mean_cost_per_ppid * total_expected_n_ppid_trials
    print(
        f"Total expected costs (based on triplets) ({'€' if in_euro else '£'}): "
        f"{total_expected_costs_triplets:.2f}"
    )
    for sess in params.SESSIONS:
        session_df.loc[sess, "total_expected_costs_triplets"] = (
            session_df.loc[sess, "total_expected_n_ppid_triplets"] * session_df.loc[sess, "mean_costs_per_ppid"]
        )
        print(f"• {sess}: {session_df.loc[sess, 'total_expected_costs_triplets']:.2f}")

    print(f"Total expected costs (based on trials) ({'€' if in_euro else '£'}): {total_expected_costs_trials:.2f}")
    for sess in params.SESSIONS:
        session_df.loc[sess, "total_expected_costs_trials"] = (
            session_df.loc[sess, "total_expected_n_ppid_trials"] * session_df.loc[sess, "mean_costs_per_ppid"]
        )
        print(f"• {sess}: {session_df.loc[sess, 'total_expected_costs_trials']:.2f}")

    cprint(
        string=f"Remaining expected costs (based on triplets) ({'€' if in_euro else '£'}): "
        f"{expected_remaining_n_ppids_triplets * mean_cost_per_ppid:.2f}",
        col="r",
        fm="bo",
    )
    cprint(
        string=f"Remaining expected costs (based on trials) ({'€' if in_euro else '£'}): "
        f"{total_expected_costs_trials - paid_costs:.2f}",
        col="y",
        fm="bo",
    )
    for sess in params.SESSIONS:
        expected_remaining_costs_triplets = round(
            session_df.loc[sess, "expected_remaining_n_ppids_triplets"] * session_df.loc[sess, "mean_costs_per_ppid"],
            2,
        )
        session_df.loc[sess, "expected_remaining_costs_triplets"] = expected_remaining_costs_triplets
        cprint(string=f"• {sess}: {expected_remaining_costs_triplets:.2f}", col="r")

        expected_remaining_costs_trials = round(
            session_df.loc[sess, "expected_remaining_n_ppids_trials"] * session_df.loc[sess, "mean_costs_per_ppid"], 2
        )
        session_df.loc[sess, "expected_remaining_costs_trials"] = expected_remaining_costs_trials
        cprint(string=f"• {sess}: {expected_remaining_costs_trials:.2f}", col="y")

    print()
    print("o*+*" * 20)
    print("-" * 80)
    print("o*+*" * 20)

    # Save
    session_df.to_csv(path_to_save)

    return session_df


def determine_threshold_for_minimal_response_time(session: str, plot: bool = True, verbose: bool = True) -> None:
    """Determine the minimum response time for the given session."""
    tr_table = read_trial_results_of_session(session=session, clean_trials=True, verbose=verbose)

    if verbose:
        print(tr_table.response_time.describe())

    rt_min = tr_table.response_time.min()
    rt_max = tr_table.response_time.max()
    rt_mean = tr_table.response_time.mean()
    rt_median = tr_table.response_time.median()
    rt_std = tr_table.response_time.std()
    rt_quant_0_001 = tr_table.response_time.quantile(0.001)
    rt_quant_0_002 = tr_table.response_time.quantile(0.002)

    if plot:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), num=f"Response time distribution {session}")  # fig = _
        h = tr_table.response_time.hist(bins=10 * 10, ax=ax)
        y_max = h.get_ylim()[1]
        plt.vlines(x=rt_min, ymin=0, ymax=y_max, color="r")
        plt.text(x=rt_min, y=y_max, s=f"min: {rt_min:.2f}s", color="r", rotation=45)

        plt.vlines(x=rt_mean, ymin=0, ymax=y_max, color="r")
        plt.text(x=rt_mean, y=y_max, s=f"mean: {rt_mean:.2f}s", color="r", rotation=45)

        plt.vlines(x=rt_median, ymin=0, ymax=y_max, color="orange")
        plt.text(x=rt_median, y=y_max, s=f"median: {rt_median:.2f}s", color="orange", rotation=45)

        plt.vlines(x=rt_max, ymin=0, ymax=y_max, color="r")
        plt.text(x=rt_max, y=y_max, s=f"max: {rt_max:.2f}s", color="r", rotation=45)

        plt.vlines(x=rt_mean - rt_std, ymin=0, ymax=y_max, color="r", linestyle="--", alpha=0.5)
        plt.text(x=rt_mean - rt_std, y=y_max, s=f"mean-std: {rt_mean - rt_std:.2f}s", color="r", rotation=45)

        plt.vlines(x=rt_mean + rt_std, ymin=0, ymax=y_max, color="r", linestyle="--", alpha=0.5)
        plt.text(x=rt_mean + rt_std, y=y_max, s=f"mean+std: {rt_mean + rt_std:.2f}s", color="r", rotation=45)

        plt.vlines(x=rt_quant_0_001, ymin=0, ymax=y_max, color="pink", linestyle="--", alpha=1.0)
        plt.text(x=rt_quant_0_001, y=y_max, s=f"0.1% quantile: {rt_quant_0_001:.2f}s", color="pink", rotation=45)

        plt.vlines(x=rt_quant_0_002, ymin=0, ymax=y_max, color="orchid", linestyle="--", alpha=1.0)
        plt.text(x=rt_quant_0_002, y=y_max, s=f"0.2% quantile: {rt_quant_0_002:.2f}s", color="orchid", rotation=45)
        h.set(title=f"Response time distribution for session {session} (cleaned data)")
        h.get_figure().tight_layout()


def determine_optimal_set_of_triplets_for_resampling() -> None:
    """
    Determine an optimal set of triplets for resampling.

    The following points are desired:
    1. to sample some heads / triplets multiple times (e.g., 10 times) to get an estimate of the noise in
       terms of human similarity judgments
    2. to employ triplets that were sampled already multiple times.
    """
    n_triplets = np.math.comb(params.main.n_faces, 3)
    triplet_tab = get_triplet_ids_and_heads(pilot=False)
    assert len(triplet_tab) == n_triplets, f"Expected {n_triplets} triplets, but got {len(triplet_tab)}"  # noqa: S101

    triplets_dict = {}
    for sess in params.SESSIONS:
        tr_table = read_trial_results_of_session(session=sess, clean_trials=True, verbose=False)[
            [
                "triplet_id",
                # 'head1', 'head2', 'head3',
                "triplet",
                "head_odd",
            ]
        ]
        tr_table = tr_table.astype({"triplet_id": int, "head_odd": int})
        # tr_table = tr_table.astype({'triplet_id': int, 'head1': int, 'head2': int, 'head3': int})  # noqa: ERA001

        tripl_val_ctn_tab = (
            tr_table.value_counts()
            .rename_axis(["triplet_id", "triplet", "head_odd"])
            .reset_index(name="counts")[["triplet_id", "triplet", "counts"]]
        )

        # Compute the rate of consistency in the human similarity judgments within triplets
        for _i, multi_triplet_tr in tr_table[
            tr_table.triplet_id.isin(
                tripl_val_ctn_tab[  # take triplets sampled multiple times
                    tripl_val_ctn_tab.counts > 2  # noqa: PLR2004
                ].triplet_id
            )
        ].groupby("triplet_id"):
            print(multi_triplet_tr)
            multi_triplet_tr.head_odd.nunique()
            # TODO: how to compute? continue here ...  # noqa: FIX002
            break

        # Check whether all triplets were sampled at least once
        assert (  # noqa: S101
            len(tripl_val_ctn_tab) == n_triplets
        ), f"In {sess} session not all triplets were sampled at least once!"

        triplets_dict[sess] = tripl_val_ctn_tab

    # Check which heads were sampled the most
    # TODO: continue here  # noqa: FIX002
    tv = triplets_dict["2D"]
    tv["heads"] = tv.triplet.map(lambda x: x.split("_"))
    heads = []
    for _i, row in tv.iterrows():
        heads.extend(row.heads * row.counts)
    heads = np.array(heads).astype(int)
    print(pd.value_counts(heads).describe()[["min", "max"]])


def main():
    """Run the main function to check response data."""
    # Check data of given Set
    cprint(string="\n" + "*" * 26 + f"\nChecking data of 'Set{FLAGS.set_nr}':\n" + "*" * 26 + "\n", col="b", fm="ul")

    # Identify potential low quality data
    # read trial results table
    tr_table = read_trial_results_of_set(set_nr=FLAGS.set_nr, clean_trials=False, verbose=True)
    # p_table, p2_p_table = read_prolific_participant_data(set_nr=FLAGS.set_nr, return_path=True)  # noqa: ERA001

    # Check how many subjects have missed at least one catch trial
    percentage_missed_n_catch_trials(trial_results_table=tr_table, n=1)

    if FLAGS.plot_catch:
        plot_catch_counts(trial_results_table=tr_table, set_nr=FLAGS.set_nr, save=True)

    if FLAGS.plot_rt:
        plot_response_time_distribution(trial_results_table=tr_table, compare_pilot=True, set_nr=FLAGS.set_nr)

    # Review participants
    if FLAGS.review:
        # Compute quality checks and save them in a table
        cprint(string="Participants bad quality score (BQS):", col="y", fm="ul")
        print(
            bqs_table := get_quality_score_table(
                set_nr=FLAGS.set_nr, save=FLAGS.save_bqs_table, verbose=FLAGS.save_bqs_table
            )
        )

        cprint(
            string=f"\nExplore {len(bqs_table.loc[bqs_table.BQS >= BQS_TH].ppid)} single participants with "
            f"very high 'bad quality scores' (BQS):",
            col="b",
            fm="bo",
        )

        # Review participants with high BQS
        review_prolific_participants(
            set_nr=FLAGS.set_nr,
            ppids_to_review=bqs_table.loc[bqs_table.BQS >= BQS_TH].ppid.to_list(),
            plot_rt=FLAGS.plot_rt,
            accept_at_bqs=None,
        )

        # Review single participants with low BQS (i.e., good quality)
        cprint(
            string=f"\nExplore {len(bqs_table.loc[bqs_table.BQS < BQS_TH].ppid)} single participants with "
            f"low quality scores (BQS):",
            col="b",
            fm="bo",
        )
        review_prolific_participants(
            set_nr=FLAGS.set_nr,
            ppids_to_review=bqs_table.loc[bqs_table.BQS < BQS_TH].ppid.to_list(),
            plot_rt=False,
            accept_at_bqs=ACCEPT_AT_BQS,
        )

        # Review hand selected participants
        review_prolific_participants(set_nr=FLAGS.set_nr, plot_rt=True, force_review=True, ppids_to_review=[])

        # Review participants which are left over (usually ppids w/o trial data)
        p_table = read_prolific_participant_data(set_nr=FLAGS.set_nr)

        ppids_wo_review = p_table[(p_table.Status == "AWAITING REVIEW") & (p_table.decision.isna())]["Participant id"]
        cprint(string=f"\nFollowing {len(ppids_wo_review)} participant(s) are left over to review:", col="b", fm="bo")
        print(ppids_wo_review, "\n")

        review_prolific_participants(
            set_nr=FLAGS.set_nr, plot_rt=True, force_review=True, ppids_to_review=ppids_wo_review.to_list()
        )

        # Compute comma separated (or new line) list of ppid to accept on Prolific
        print_ppid_list_to_accept(set_nr=FLAGS.set_nr)

        # Compile list for bonus payment
        min_bonus = 0.0
        if min_bonus < params.BONUS:
            print_ppid_list_for_bonus(trial_results_table=tr_table, set_nr=FLAGS.set_nr, bonus=params.BONUS)

        # Compile blocklist
        print_ppid_list_to_block(set_nr=FLAGS.set_nr)

    # Compute expected costs
    if FLAGS.costs:
        estimate_remaining_costs(in_euro=False, verbose=True)
        if not FLAGS.triplet_table:
            finalized_triplets(session=FLAGS.set_nr[0] + "D")

    if FLAGS.triplet_table:
        current_session = FLAGS.set_nr[0] + "D"
        update_triplet_table_on_dynamodb(
            session=current_session,
            set_finalised_triplets_to_g=False,  # toggle manually if requested
            delete_done_triplets=False,
        )


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Add arg parser
    parser = argparse.ArgumentParser(description="Check data of given Set")  # init argument parser

    # Review arguments
    parser.add_argument(
        "-s",
        "--set_nr",
        type=str,
        action="store",
        default=SET_NUMBER,
        help="Define Set of 2D session: (2.0, 2.1, ...) OR 3D session: (3.0, 3.1, ...)",
    )
    parser.add_argument(
        "--save_bqs_table",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save the BQS table (mainly do it when new data arrives)",
    )
    parser.add_argument(
        "-r", "--review", action=argparse.BooleanOptionalAction, default=True, help="Review participants"
    )
    parser.add_argument(
        "-rt", "--plot_rt", action=argparse.BooleanOptionalAction, default=True, help="Plot response times"
    )
    parser.add_argument(
        "-catch", "--plot_catch", action=argparse.BooleanOptionalAction, default=True, help="Plot catch trials"
    )
    parser.add_argument(
        "-c", "--costs", action=argparse.BooleanOptionalAction, default=False, help="Estimate remaining costs"
    )
    parser.add_argument(
        "-td",
        "--triplet_table",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Update triplet table on DynamoDB based on remaining triplets in session",
    )

    # Arguments & parameters for BQS estimates
    parser.add_argument(
        "--bqs_th",
        type=float,
        default=BQS_TH,
        help="Bad quality score (bqs) threshold (higher than this is considered bad)",
    )
    parser.add_argument(
        "--accept_at_bqs",
        type=int,
        default=ACCEPT_AT_BQS,
        help="BQS threshold for automatic acceptance of participants",
    )
    parser.add_argument("--n_train", type=int, default=params.n_train_trials, help="Number of training trials")
    parser.add_argument("--th_mono", type=int, default=TH_MONO, help="Threshold for monotonous choices of same side")
    parser.add_argument(
        "--th_mono_rt",
        type=float,
        default=TH_MONO_RT,
        help="Threshold for response time (responses longer than this are not considered "
        "as monotonous behavior, which we consider as behavior of inattentive "
        "participants, who want to save time)",
    )
    parser.add_argument(
        "--th_rt",
        type=float,
        default=TH_RT,
        help="Threshold (in sec) for rapid response times (below is considered rapid)",
    )
    parser.add_argument("--w_catch", type=float, default=W_CATCH, help="BQS weight for catch trials")
    parser.add_argument("--w_mono", type=float, default=W_MONO, help="BQS weight for monotonous choices")
    parser.add_argument(
        "--th_mono_fast_rt",
        type=float,
        default=TH_MONO_FAST_RT,
        help="Threshold for fast response time during monotonous choice behavior",
    )
    parser.add_argument(
        "--w_mono_rt",
        type=float,
        default=W_MONO_RT,
        help="BQS weight for fast response time during monotonous choice behavior",
    )
    parser.add_argument("--w_to", type=float, default=W_TO, help="BQS weight for timeouts (i.e., no choice made)")
    parser.add_argument("--w_rt", type=float, default=W_RT, help="BQS weight for response time")
    parser.add_argument(
        "--th_consec_rt",
        type=int,
        default=TH_CONSEC_RT,
        help="Threshold for rapid response times that happen consecutively",
    )
    parser.add_argument(
        "--w_consec_rt",
        type=float,
        default=W_CONSEC_RT,
        help="BQS weight for consecutive rapid response times (added on top of RT BQS)",
    )

    # Parse arguments
    FLAGS, unparsed = parser.parse_known_args()

    # %% Run main
    main()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
