# !/usr/bin/env python3
"""
Read & prepare data files.

This module can be run from the command line interface (CLI) to read and prepare data for further analysis.

!!! tip "CLI Usage"
    ```bash
    python -m facesim3d.read_data --help
    ```

With this, one can also delete the processed data from the remote (`DynamoDB`).
"""

# %% Import
from __future__ import annotations

import argparse
import os
import warnings
from ast import literal_eval
from datetime import datetime
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from string import ascii_lowercase

warnings.simplefilter(action="ignore", category=FutureWarning)  # issue with pandas 1.5.1

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from ut.ils import ask_true_false, browse_files, cinput, cprint, tree

from facesim3d.configs import params, paths

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

SET_NUMBER: str = "3.20"  # define 2D session: (2.0, 2.1, ...) OR 3D session: (3.0, 3.1, ...)

COMPLETION_CODES = {
    "completed": {
        "2.0": "CX2Q37VH",
        "2.1": "CX2Q37VH",
        "2.2": "CX2Q37VH",
        "2.3": "CX2Q37VH",
        "2.4": "CX2Q37VH",
        "2.5": "CX2Q37VH",
        "2.6": "CX2Q37VH",
        "2.7": "CX2Q37VH",
        "2.8": "CX2Q37VH",
        "2.9": "CX2Q37VH",
        "2.10": "CX2Q37VH",
        "2.11": "CX2Q37VH",
        "2.12": "CX2Q37VH",
        "2.20": "CX2Q37VH",
        "3.0": "CX2Q37VH",
        "3.1": "CX2Q37VH",
        "3.2": "CX2Q37VH",
        "3.3": "CX2Q37VH",
        "3.4": "CX2Q37VH",
        "3.5": "CX2Q37VH",
        "3.6": "CX2Q37VH",
        "3.7": "CX2Q37VH",
        "3.8": "CX2Q37VH",
        "3.9": "CX2Q37VH",
        "3.10": "CX2Q37VH",
        "3.20": "CX2Q37VH",
    },
    "failed": {
        "2.0": "CMV7ODK8",
        "2.1": "CMV7ODK8",
        "2.2": "CMV7ODK8",
        "2.3": "CMV7ODK8",
        "2.4": "CMV7ODK8",
        "2.5": "CMV7ODK8",
        "2.6": "CMV7ODK8",
        "2.7": "C1FX264D",
        "2.8": "C1FX264D",
        "2.9": "C1FX264D",
        "2.10": "C1FX264D",
        "2.11": "C1FX264D",
        "2.12": "C1FX264D",
        "2.20": "C1FX264D",
        "3.0": "CMV7ODK8",
        "3.1": "CMV7ODK8",
        "3.2": "CMV7ODK8",
        "3.3": "CMV7ODK8",
        "3.4": "C1FX264D",
        "3.5": "C1FX264D",
        "3.6": "C1FX264D",
        "3.7": "C1FX264D",
        "3.8": "C1FX264D",
        "3.9": "C1FX264D",
        "3.10": "C1FX264D",
    },
    "completed2.0": {
        "2.7": "C1MT6VRN",
        "2.8": "C1MT6VRN",
        "2.9": "C1MT6VRN",
        "2.10": "C1MT6VRN",
        "2.11": "C1MT6VRN",
        "2.12": "C1MT6VRN",
        "2.20": "C1MT6VRN",
        "3.4": "C1MT6VRN",
        "3.5": "C1MT6VRN",
        "3.6": "C1MT6VRN",
        "3.7": "C1MT6VRN",
        "3.8": "C1MT6VRN",
        "3.9": "C1MT6VRN",
        "3.10": "C1MT6VRN",
        "3.20": "C1MT6VRN",
    },
}

# Pilot dropouts
dropouts_pilot_v2 = [
    # NOTE: DATA REMOVED
]  # -> define criteria quantitatively

# Relevant data columns
SORTED_COLS_PILOT = [
    "ppid",  # participant ID
    "trial_num",  # trial number in full participant session
    "block_num",  # number of a block
    "trial_num_in_block",  # trial number within a block
    "break_time",  # time of break after a block (always at 'trial_num_in_block' == 60)
    "start_time",  # start time of a trial during full participant session
    "end_time",  # end time of a trial during full participant session
    "response_time",  # response time in a trial of a participant
    "triplet_id",  # ID of face triplet
    "correct",  # whether head was chosen or not (i.e., time passed)
    "head1",  # left head
    "head2",  # middle head
    "head3",  # right head
    "head_odd",  # head chosen by participant to be the odd-one-out (0: none was chosen)
    "catch_head",  # 0 if no catch trial, otherwise ID/number of the head which was duplicated to
    # create catch trial. The duplicate replaces a random other head in the triplet
    "caught",  # always False, but if a participant does not notice the catch trial, it is set to True
]
# For main experiment
SORTED_COLS = [
    *SORTED_COLS_PILOT,
    "triplet",  # head-numbers of a triplet (sorted numerically)
    "session_num",  # session number (0: prolific pilot, 1-3: main)
    "keyPress",  # key (side) pressed by participant
    "SystemDateTime_BeginTrial",  # Start date-time of one trial
    "ppid_session_dataname",
]  # ppid + session
# ["experiment"]  # removed columns  # noqa: ERA001

# DynamoDB-conversion map of data types
DT_MAP = {"S": str, "N": float, "BOOL": bool}


# %% Pilot specific functions ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def read_pilot_participant_data() -> pd.DataFrame:
    """
    Get the full participant table of the pilot study (version 2).

    :return: participant table
    """
    # Load full table
    participant_files = list(Path(paths.data.pilot.v2).glob("*UXFData.FaceSim.ParticipantDetails.csv"))
    if len(participant_files) > 1:
        participant_files = browse_files(initialdir=paths.data.pilot.v2, filetypes="*.csv")
    else:
        participant_files = participant_files.pop()

    # Check if the full (concatenated) table is already there
    participant_full_table_path = Path(
        str(participant_files).replace("ParticipantDetails", "ParticipantDetails_processed")
    )
    if participant_full_table_path.is_file():
        table_processed = pd.read_csv(participant_full_table_path, index_col=0)

    else:
        table_processed = pd.read_csv(participant_files)

        # Exchange ppid_session_dataname with ppid
        table_processed = table_processed.rename(columns={"ppid_session_dataname": "ppid"})
        table_processed.ppid = table_processed.ppid.replace("_s001_participant_details", "", regex=True)

        # Remove UnlockTriplets user (has NaN's in 'group_exp' column)
        table_processed = table_processed[table_processed.ppid != "UnlockTriplets"].reset_index(drop=True)
        # table_processed.drop(index=table_processed[table_processed.group_exp.isna()].index, axis=1,
        #                      inplace=True)  # This should be the UnlockTriplets 'User'

        # Clean up columns
        dtype_map = {"N": np.int64, "S": str}  # "D": np.datetime64
        for col in table_processed.columns:
            if col == "ppid":
                continue

            # Get dtype
            cell = table_processed[col].iloc[0]
            dt = cell[cell.find(":") - 2 : cell.find(":") - 1]
            # Update cells in column
            table_processed[col] = (
                table_processed[col]
                .map(lambda x: x.replace('[{"' + f'{dt}":"', ""))  # noqa: B023
                .replace('"}]', "", regex=True)
                .astype(dtype_map[dt])
            )

        # Save processed table
        table_processed.to_csv(participant_full_table_path)

    return table_processed


def read_pilot_data(clean_trials: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Get the full trial table of the pilot study (version 2).

    This table is downloaded as `csv` in one sweep from `DynamoDB`.

    :param clean_trials: clean trials (remove trials with no response, etc.)
    :param verbose: be verbose or not
    :return: processed trial table
    """
    # Check data dir
    if verbose:
        tree(paths.data.pilot.v2)

    # Load full table
    trial_result_files = list(Path(paths.data.pilot.v2).glob("*UXFData.FaceSim.TrialResults.csv"))
    if len(trial_result_files) > 1:
        trial_result_files = browse_files(initialdir=paths.data.pilot.v2, filetypes="*.csv")
    else:
        trial_result_files = trial_result_files.pop()

    # Check if the full (concatenated) table is already there
    trial_result_full_table_path = str(trial_result_files).replace("TrialResults", "TrialResults_processed")
    if Path(trial_result_full_table_path).is_file():
        table_processed = pd.read_csv(trial_result_full_table_path)

    else:
        table_processed = pd.read_csv(trial_result_files)

    # Remove unnecessary columns & sort the rest
    table_processed = table_processed[SORTED_COLS_PILOT]

    # Sort rows
    table_processed = table_processed.sort_values(
        by=["ppid", "trial_num"], axis=0, ascending=True, inplace=False
    ).reset_index(drop=True)

    # Save table
    table_processed.to_csv(trial_result_full_table_path, index=False)

    if clean_trials:
        cath_head_trials = 0.0
        table_processed = table_processed[table_processed.block_num > 1]  # remove training
        table_processed = table_processed[table_processed.catch_head == cath_head_trials]  # remove catch trials
        table_processed = table_processed[table_processed.head_odd != 0]  # remove time-outs
        table_processed = table_processed[~table_processed.ppid.isin(dropouts_pilot_v2)]

    return table_processed.reset_index(drop=True)


def _read_pilot_results_json_data_via_s3(verbose: bool = False) -> pd.DataFrame:
    """
    Get the full trial table of the pilot study (version 2) from memory.

    This table must be constructed from several tables downloaded from `S3`.

    :param verbose: Be verbose or not
    :return: processed Trial table
    """
    # Define the session path
    p2_pilot_s3 = Path(paths.data.pilot.v2, "via_s3")

    if verbose:
        tree(p2_pilot_s3)

    # Find the correct directory
    trial_result_dirs = [d for d in os.listdir(p2_pilot_s3) if (d.startswith("2022-") and not d.endswith(".csv"))]

    if len(trial_result_dirs) > 1:
        trial_result_dirs = Path(browse_files(initialdir=p2_pilot_s3, filetypes=".json")).parent
    else:
        trial_result_dirs = trial_result_dirs.pop()
    trial_result_dirs = Path(p2_pilot_s3, trial_result_dirs)

    # Set path to processed trial table
    trial_result_full_table_path = Path(str(trial_result_dirs) + ".csv")

    # Check if the full (concatenated) table is already there
    if trial_result_full_table_path.is_file():
        table_processed = pd.read_csv(trial_result_full_table_path)

    else:
        # Read json files
        trial_result_files = [f for f in os.listdir(trial_result_dirs) if f.endswith(".json")]
        table_processed = None  # one table to save all data
        for json_file_path in trial_result_files:
            p2_json = trial_result_dirs / json_file_path

            # Process json file via pandas
            table_raw = pd.read_json(p2_json, lines=True)

            # Remove type info from json file
            for row in table_raw.values:  # noqa: PD011
                current_row = row.item()
                # pprint(current_row)  # from pprint import pprint  # noqa: ERA001

                trial_tab = pd.DataFrame(current_row)
                # append empty row to table
                copy_row = trial_tab.iloc[0:1].copy()
                copy_row[~pd.isna(copy_row)] = np.nan
                trial_tab = pd.concat([trial_tab, copy_row])

                # Write non-nan-value in empty row for each column
                for col in trial_tab.columns:
                    trial_tab.iloc[-1][col] = trial_tab[col].dropna().item()

                # Keep only filled row
                trial_tab = trial_tab.iloc[-1:]

                # Concatenate to big table
                table_processed = trial_tab if table_processed is None else pd.concat([table_processed, trial_tab])
                # , ignore_index=True)

        cprint(str({type(val) for val in table_processed["triplet_id"].values}), col="r")  # noqa: PD011
        table_processed.set_index("triplet_id", inplace=False).sort_index().head(15)
        cprint(
            str(
                [idx for idx in table_processed.set_index("triplet_id", inplace=False).index if isinstance(idx, float)]
            ),
            col="r",
        )

        # Fill empty slots with nan
        table_processed = table_processed.replace("", np.nan)

        # Remove unnecessary columns & sort the rest
        table_processed = table_processed[SORTED_COLS_PILOT]

        # Adapt dtypes
        new_dtypes = [
            str,
            int,
            int,
            int,
            float,
            float,
            float,
            float,
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]  # same order as SORTED_COLS_PILOT
        table_processed = table_processed.astype(dict(zip(SORTED_COLS_PILOT, new_dtypes, strict=True)))

        # Sort rows
        table_processed = table_processed.sort_values(
            by=["ppid", "trial_num"], axis=0, ascending=True, inplace=False
        ).reset_index(drop=True)

        # Save table
        table_processed.to_csv(trial_result_full_table_path, index=False)

    return table_processed


# %% Main Functions o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def create_all_triple_combinations(n_faces: int) -> pd.DataFrame:
    """Create all triplet combinations."""
    n_faces_in_triplet: int = 3
    if n_faces < n_faces_in_triplet:
        msg = f"Number of faces must be at least 3, but is {n_faces}!"
        raise ValueError(msg)
    triplet_combinations = list(combinations(range(1, n_faces + 1), r=n_faces_in_triplet))
    return pd.DataFrame(triplet_combinations, columns=["head1", "head2", "head3"])


def set_infix(set_nr: str) -> str:
    """Generate the Set infix (e.g., 's004' OR 's011') from a set number."""
    return f"s{int(set_nr.split('.')[-1]):03d}"


def get_triplet_ids_and_heads(pilot: bool = params.PILOT) -> pd.DataFrame:
    """
    Get all `triplet_id`'s and heads corresponding to `UXFData.FaceSim.TripletsIDB.*D[.Pilot]`.

    Of the form:

        triplet_id   triplet
                 1  19_26_56
                 2  22_68_73
                 3  35_39_54
               ...       ...

    :param pilot: Whether to load pilot data or not (main).
    :return: Table of the triplet_ids and heads [`pd.DataFrame`].
    """
    p2_table = paths.data.pilot.triplets if pilot else paths.data.main.triplets
    return pd.read_csv(p2_table)


def get_list_of_acquired_sets() -> list:
    """Get the list of all sets that have been acquired."""
    return sorted(
        [str(f).split("-Set")[1].split("_")[0] for f in Path(paths.data.main.prolific).glob("*Participants-Set*.csv")]
    )  # [2.0, 2.1, ..., 3.0, 3.1,  ...]


def load_local_table(table_name: str | None = None) -> pd.DataFrame | None:
    """Load a `UXFData.FaceSim.*` table from the local storage system."""
    if table_name is None:
        cprint(string="Specify the table to load:", col="y", fm="ul")
        p2_table = browse_files(initialdir=paths.data.MAIN, filetypes="*.csv")
    else:
        p2_table = list(Path(paths.data.MAIN).glob(f"*{table_name}*"))
        if len(p2_table) > 1:
            cprint(
                string=f"Found more than one file w.r.t. table '{table_name}'!\n"
                f"Choose the corresponding table file by index:",
                col="b",
            )
            print("", *[f"{i}:\t'{tab.name}'" for i, tab in enumerate(p2_table)], sep="\n\t")
            tab_idx = cinput(string="\nType index of table you want to load: ", col="y")
            p2_table = p2_table[int(tab_idx)]
        elif len(p2_table) == 0:
            cprint(string=f"No table found w.r.t. '{table_name}'!", col="y")
            return None
        else:
            p2_table = p2_table.pop()

    # Load tablet
    tab = pd.read_csv(p2_table)

    # Convert table rows if necessary: unpack the DynamoDB json format
    for col in tab.columns:
        if isinstance(tab[col][0], str) and tab[col][0].startswith("[{"):
            tab[col] = tab[col].map(literal_eval)

            if isinstance(tab[col][0], list) and len(tab[col][0]) == 1:
                tab[col] = tab[col].map(lambda x: x[0])
                tab[col] = convert_dynamodb_json_in_row_of_df(df_row=tab[col])
            elif isinstance(tab[col][0], list) and len(tab[col][0]) > 1:
                # Primarily a case for 'UXFData.FaceSim.SessionLog'.
                # Most cells in row (i.e., one participant session)
                # are lists of DynamoDB jsons (i.e., dicts)
                # convert [{'S': 'Log'}, {'S': 'Log'}, , ...] -> [Log, Log, ...] per row
                tab[col] = tab[col].map(lambda x: [cell[next(iter(cell.keys()))] for cell in x])
            else:
                cprint(string=f"Unknown format of column '{col}' at index 0!", col="r")

        if isinstance(tab[col][0], str) and tab[col][0].startswith("["):
            # Primarily a case for 'UXFData.FaceSim.SessionLog' after formatting
            tab[col] = tab[col].map(literal_eval)

        if "SystemDateTime" in col:
            tab[col] = tab[col].map(convert_date_time)

    return tab


def read_participant_data(process: bool = False) -> pd.DataFrame:
    """
    Get the full participant table of the main study.

    Table name: `'*_UXFData.FaceSim.ParticipantDetails_processed.csv'`.

    :param process: True: force (re-)processing of data
    :return: participant table
    """
    # Load full table
    p2_participant_files = list(Path(paths.data.MAIN).glob("*UXFData.FaceSim.ParticipantDetails.csv"))

    if len(p2_participant_files) > 1:
        p2_participant_files = Path(browse_files(initialdir=paths.data.MAIN, filetypes="*.csv"))
    else:
        p2_participant_files = p2_participant_files.pop()

    p2_raw_participant_files = Path(
        str(p2_participant_files).replace(paths.data.MAIN, paths.data.main.archive).replace(".csv", "_raw.csv")
    )

    # Check if the full (concatenated) table is already there
    if p2_raw_participant_files.exists() and not process:
        # We know that p2_participant_files.is_file() is True
        table_processed = pd.read_csv(p2_participant_files)

    else:  # process == True:
        table_processed = load_local_table(table_name=p2_participant_files.name)

        # Extract from ppid_session_dataname the ppid
        if "ppid" not in table_processed.columns:
            table_processed["ppid"] = table_processed.ppid_session_dataname.map(lambda x: x.split("_s0")[0])

        # Remove debug & UnlockTriplets (has NaN's in 'group_exp' column) users
        table_processed = table_processed.loc[
            table_processed.ppid.isin([p for p in table_processed.ppid if "debug" not in p])
        ]
        table_processed = table_processed[table_processed.ppid != "UnlockTriplets"].reset_index(drop=True)

        # Archive unprocessed table
        p2_participant_files.rename(p2_raw_participant_files)

        # Save processed table
        table_processed.to_csv(p2_participant_files, index=False)

    return table_processed


def read_prolific_participant_data(
    set_nr: str | float, return_path: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, str]:
    """
    Read the participant table of a given Set downloaded from Prolific.

    :param set_nr: Prolific Set number 2.* for 2D AND 3.* for 3D
    :param return_path: if True also return the path to the file
    :returns: participant table of the given Prolific Set
    """
    p_files = [
        f for f in os.listdir(paths.data.main.prolific) if f.endswith(".csv") and f"Participants-Set{set_nr}_" in f
    ]
    if not p_files:  # empty
        cprint(string=f"No participant table found for Set{set_nr}!", col="r")
        return None
    if len(p_files) > 1:
        cprint(string=f"Choose the corresponding Participant file of Set{set_nr}!", col="b")
        cprint(string="Note: There should be only one file per Set!", col="y")
        p_files = browse_files(initialdir=paths.data.main.prolific, filetypes="*.csv")
    else:
        p_files = p_files.pop()

    # Read table
    full_path = Path(paths.data.main.prolific, p_files)
    ppid_prolific_table = pd.read_csv(full_path)

    # Add decision column (if not there)
    if "decision" not in ppid_prolific_table.columns:
        ppid_prolific_table["decision"] = np.nan

    if return_path:
        return ppid_prolific_table, str(full_path)
    return ppid_prolific_table


def get_participant_session(ppid: str, pilot: bool = params.PILOT) -> str | None:
    """
    Get session (2D, 3D) of the given participant ID (`ppid`).

    !!! note
        Participant can only be part of one session (2D or 3D).

    :param ppid: ID of participant
    :param pilot: True: use pilot data
    :return: session of participant
    """
    # Get participant table
    pid_table = read_pilot_participant_data() if pilot else read_participant_data()
    session = pid_table[pid_table.ppid == ppid].group_exp
    session = session.drop_duplicates()
    if len(session) == 1:
        return session.item()  # extract session
    if len(session) > 1:
        msg = f"Participant '{ppid}' was part of different conditions!\n{session}\nThis must be solved manually!"
        raise ValueError(msg)
    cprint(string=f"Participant '{ppid}' not found!", col="r")
    return None


def get_participant_set_numbers(ppid: str) -> list[str]:
    """
    Get the Set-number(s) of a given participant.

    !!! note
        Participants can be part of up to three sets, however, only of one session (2D or 3D).

    """
    set_nrs = []
    for set_nr in get_list_of_acquired_sets():
        prolific_tab = read_prolific_participant_data(set_nr=set_nr)
        if ppid in prolific_tab["Participant id"].values:  # noqa: PD011
            set_nrs.append(set_nr)
    return set_nrs


def convert_date_time(date_time: str) -> str:
    """Convert date-time string to format `YYYY-MM-DD HH:MM:SS:MS`."""
    if pd.isna(date_time):
        return date_time
    if not date_time.startswith("2022-") and not date_time.startswith("2023-"):
        # Bring in format YYYY-MM-DD HH:MM:SS:MS (e.g., 2022-11-14 17:12:18:358)
        d = date_time[: date_time.find(":") - 2]
        t = date_time[date_time.find(":") - 2 :]
        d = d.replace(" ", "").replace(".", "-").replace("/", "-")  # remove blanks & replace
        date_time = f"{d} {t}"
    if date_time[10] != " ":
        # This solves an issue with dates like this '2022-12-1412-19-28- 880'
        date_time = date_time[:10] + " " + date_time[10:].replace("-", ":").replace(": ", ".")

    return date_time


def merge_tables(df: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, bool]:
    """Merge a given table (`df`) with an existing table of the given table name."""
    cprint(string="Merging tables ...", col="b")
    merge = True  # init
    df2 = load_local_table(table_name=table_name)
    if df2 is None:
        merge = False
        merged_df = df
    else:
        if set(df.columns) != set(df2.columns):
            msg = "No column match of downloaded & local tables!"
            raise ValueError(msg)
        merged_df = pd.concat([df2, df], ignore_index=True)

        if merged_df.duplicated().any():
            cprint(string=f"Dropping {merged_df.duplicated().sum()} duplicates ...", col="b")
            merged_df = merged_df.drop_duplicates(ignore_index=True)

    return merged_df, merge


def archive_former_tables(path_to_save: str | Path, table_name: str) -> None:
    """
    Move former tables to archive.

    :param path_to_save: Path where new table will be saved
    :param table_name: name of table
    """
    list_of_other_tables = Path(str(path_to_save)).parent.glob(f"*{table_name}.csv")
    list_of_other_tables = [p for p in list_of_other_tables if str(p) != path_to_save]

    for p in list_of_other_tables:
        p.rename(str(p).replace(paths.data.MAIN, paths.data.main.archive))


def load_table_from_dynamodb(table_name: str | None = None, save: bool = False, merge: bool = False) -> pd.DataFrame:
    """Load a `UXFData.FaceSim.*` table from DynamoDB."""
    dynamodb = boto3.resource("dynamodb", region_name="eu-central-1")  # connect to DynamoDB

    table_list = list(dynamodb.tables.all())  # pull all tables (names) from DynamoDB
    table_list = [t.name for t in table_list]  # extract table names

    if table_name is None:
        cprint(string="Specify table to download:", col="y", fm="ul")
        print("", *[f"{i}:\t'{tab.split('.FaceSim.')[-1]}'" for i, tab in enumerate(table_list)], sep="\n\t")
        tab_idx = cinput(string="\nType index of table you want to download: ", col="y")
        table_name = table_list[int(tab_idx)]
    elif table_name not in table_list:
        msg = (
            f"Given table '{table_name}' was not found on DynamoDB!\n"
            f"\nFollowing tables are available:\n\n{table_list!s}"
        )
        raise ValueError(msg)

    cprint(string=f"Scanning & loading table '{table_name}' from DynamoDB ...", col="b")

    table = dynamodb.Table(table_name)

    response = table.scan()  # -> dict
    data = response["Items"]
    # The following is necessary because the response is paginated (limit 1 MB)
    while "LastEvaluatedKey" in response:
        response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        data.extend(response["Items"])

    # Convert to pandas DataFrame
    loaded_df = pd.DataFrame(data)

    # Unbox table cells
    # Some tables have list entries in cells, with list-length==0 (see below): Unpack them:
    #     screen_width      SystemDateTime_StartExp  ... screen_height age_years
    # 0         [1600]    [2022/11/14 15:50:05.086]  ...         [900]      [25]
    # 1         [1920]    [2022/11/14 14:51:55.245]  ...        [1080]      [21]
    # 2         [1440]    [2022/11/14 15:49:49.150]  ...         [900]      [22]

    # Filter out triplet unlocker
    ppid_sess_col = [c for c in loaded_df.columns if c.startswith("ppid_session")]
    for i, v in loaded_df[ppid_sess_col].iterrows():  # noqa: B007
        if isinstance(v.values[0], str) and v.values[0].startswith("UnlockTriplets_"):  # noqa: PD011
            break
    else:
        i = None
    if i is not None:
        loaded_df = loaded_df.drop(index=i).reset_index(drop=True)

    # Process columns
    for col in loaded_df.columns:
        rnd_idx = np.random.randint(0, len(loaded_df), 10)
        if all(
            (isinstance(cell, list) and len(cell) == 1)
            for cell in loaded_df[col].iloc[rnd_idx].values  # noqa: PD011
        ):  # noqa: PD011, RUF100
            loaded_df[col] = loaded_df[col].map(lambda x: x[0])
        # Do not unpack those with list length > 1

        # Correct datetime in table
        if "SystemDateTime" in col:
            loaded_df[col] = loaded_df[col].map(convert_date_time)

    # Extract ppid from ppid_session_dataname
    if "ppid_session_dataname" in loaded_df.columns and "ppid" not in loaded_df.columns:
        loaded_df["ppid"] = loaded_df.ppid_session_dataname.map(lambda x: x.split("_s0")[0])

    # Merge with existing tables
    log_or_trial = (
        "UXFData.FaceSim.TrialResults" in table_name
        or "UXFData.FaceSim.OtherTrialData" in table_name
        or "UXFData.FaceSim.SessionLog" in table_name
    )
    merge = merge and not log_or_trial
    # *.TrialResults & *.SessionLog tables are relatively big and should not be merged
    df_split = None  # init / this is due to the updated protocol, how data is written from May 2023
    table_name_split = None  # init
    if merge:
        if table_name == "UXFData.FaceSim.OtherSessionData":
            table_name_split = table_name.replace("OtherSessionData", "ParticipantDetails")
            df_split = loaded_df[loaded_df.ppid_session_dataname.str.contains("_participant_details")].reset_index(
                drop=True
            )
            loaded_df = loaded_df[~loaded_df.ppid_session_dataname.str.contains("_participant_details")].reset_index(
                drop=True
            )
            # Remove empty columns in dfs
            loaded_df = loaded_df.dropna(axis=1, how="all")
            df_split = df_split.dropna(axis=1, how="all")

            df_split, _ = merge_tables(df=df_split, table_name=table_name_split)

        loaded_df, merge = merge_tables(df=loaded_df, table_name=table_name)

    if table_name == "UXFData.FaceSim.TrialResults":
        # Merge 'TrialResults' with 'OtherTrialData' table
        table_name_other = table_name.replace("TrialResults", "OtherTrialData")
        df_other = load_table_from_dynamodb(table_name=table_name_other, merge=False, save=False)
        if set(loaded_df.columns) != set(df_other.columns):
            msg = "No column match of 'TrialResults' & 'OtherTrialData' tables!"
            raise ValueError(msg)
        loaded_df = pd.concat([df_other, loaded_df], ignore_index=True)

        if loaded_df.duplicated().any():
            cprint(f"Dropping {loaded_df.duplicated().sum()} duplicates ...", col="b")
            loaded_df = loaded_df.drop_duplicates(ignore_index=True)

        # Remove column 'trial_results_location_0'
        loaded_df = loaded_df.drop(columns=["trial_results_location_0"])

    if save:
        # Save table
        date_tag = str(datetime.today().date())
        path_to_save = Path(paths.data.MAIN, f"{date_tag}_{table_name}.csv")

        i = 0
        while log_or_trial and path_to_save.exists():
            # Do not overwrite *.TrialResults & *.SessionLog tables
            path_to_save = Path(str(path_to_save.replace(date_tag, f"{date_tag}{ascii_lowercase[i]}")))
            i += 1

        loaded_df.to_csv(path_to_save, index=False)
        path_to_save_split = None  # init
        if df_split is not None:
            path_to_save_split = Path(str(path_to_save.replace(table_name, table_name_split)))
            df_split.to_csv(path_to_save_split, index=False)

        if merge:
            # Move former tables to archive
            archive_former_tables(path_to_save=path_to_save, table_name=table_name)
            if path_to_save_split is not None:
                archive_former_tables(path_to_save=path_to_save_split, table_name=table_name_split)

    return loaded_df


def convert_dynamodb_json_in_row_of_df(df_row: pd.Series) -> pd.Series:
    """
    Convert a row in a pandas Dataframe (df), which is in the 'DynamoDB` `json`-format, into a normal df format.

    Rows/cells come often in the following `json` format of `DynamoDB`: `' [{"N":"25"}]'`, or similar.
    Convert this (example) to: `25`.
    """
    _row = df_row.copy()

    # Get the type and key of the type
    dtype_key = next(iter(literal_eval(_row[0])[0].keys()))
    row_dtype = DT_MAP[dtype_key]  # type mapper defined above

    # Convert row
    _row = _row.map(lambda x: literal_eval(x)[0][dtype_key])
    return _row.astype(row_dtype)


def load_trial_results_from_dynamodb(
    bucket_name: str | None = "facesimdb",
    via_s3: bool = False,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load all trial-results from `DynamoDB`.

    ??? tip "Blog posts on getting data from DynamoDB with Python and Boto3"

        https://www.fernandomc.com/posts/ten-examples-of-getting-data-from-dynamodb-with-python-and-boto3/
        https://dashbird.io/blog/aws-s3-python-tricks/

        Also, check out this `Stack Overflow` post:

        https://stackoverflow.com/questions/10450962/how-can-i-fetch-all-items-from-a-dynamodb-table-without-specifying-the-primary-k

    :return: Pandas DataFrame with all trial results
    """
    if not via_s3:
        # Get all trial results directly from DynamoDB
        return load_table_from_dynamodb(table_name="UXFData.FaceSim.TrialResults", save=save)

    # Download trial results from S3 (after export of table to S3)
    s3 = boto3.client("s3")

    if bucket_name is None:
        bucket_list = s3.list_buckets()["Buckets"]

        if len(bucket_list) > 1:
            # TODO: implement choice of bucket  # noqa: FIX002
            bucket_name = None
            msg = "More than one bucket found. Implement selection functionality!"
            raise NotImplementedError(msg)
        else:  # noqa: RET506
            bucket_name = bucket_list[0]["Name"]  # == "facesimdb"

    available_files = s3.list_objects(Bucket=bucket_name)

    available_files = [
        f for f in available_files["Contents"] if (f["Key"].endswith(".json.gz") and "AWSDynamoDB/" in f["Key"])
    ]

    if verbose:
        cprint(string=f"Found following files to download in S3 bucket '{bucket_name}':", fm="ul")
        print("", *[f"{f['LastModified']!s} : {f['Key']}" for f in available_files], sep="\n\t> ")

    # Check for different download folders
    data_folder = {d["Key"].split("/data/")[0].split("AWSDynamoDB/")[-1] for d in available_files}

    if len(data_folder) > 1:
        # In case there are multiple folders to download from, choose a folder
        data_folder = list(data_folder)
        cprint(string="Specify folder to download from:", col="y", fm="ul")
        print("", *[f"{i}:\t'{d}'" for i, d in enumerate(data_folder)], sep="\n\t")
        f_idx = cinput(string="\nType index of folder you want to download from: ", col="y")
        data_folder = data_folder[int(f_idx)]

    else:
        data_folder = data_folder.pop()

    available_files = [f for f in available_files if data_folder in f["Key"]]  # filter for folder

    for s3_file in available_files:
        # Download file
        file_date = str(s3_file["LastModified"].date())
        p2_store = Path(paths.data.main.s3, "TrialResults", file_date, s3_file["Key"].split("/")[-1])
        if p2_store.is_file():
            cprint(string=f"File '{p2_store}' already exists. Skipping download.", col="g")
            continue
        p2_store.parent.mkdir(parents=True, exist_ok=True)

        s3.download_file(Bucket=bucket_name, Key=s3_file["Key"], Filename=p2_store)
        print("Downloaded file to:", p2_store)

    cprint(string="Running now the following function: read_and_convert_s3_results_json_data() ...", col="y")
    return read_and_convert_s3_results_json_data(verbose=verbose)


def delete_all_items_in_table_on_dynamodb(table_name: str) -> None:
    """Delete all items in table on `DynamoDB` (e.g., `'UXFData.FaceSim.TrialResults'`)."""
    delete = ask_true_false(
        f"\nAre you sure you downloaded and saved all data/items of table '{table_name}' from DynamoDB?", col="r"
    )

    if delete and ask_true_false(
        f"Are you sure you want to delete all items in table '{table_name}' on DynamoDB?", col="r"
    ):
        cprint(string=f"Scanning & deleting all items in '{table_name}' on DynamoDB ...", col="y")

        dynamodb = boto3.resource("dynamodb", region_name="eu-central-1")  # connect to DynamoDB
        table = dynamodb.Table(table_name)

        response = table.scan()
        data = response["Items"]
        # The following is necessary because the response is paginated (limit 1 MB)
        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            data.extend(response["Items"])

        key_schema = table.key_schema
        key_names = [k["AttributeName"] for k in key_schema]
        with table.batch_writer() as batch:
            for row in tqdm(data, desc=f"Deleting items in {table_name}"):
                batch.delete_item(
                    Key=dict(zip(key_names, [row[key] for key in key_names], strict=True))
                    # Key={"ppid_session_dataname": row["ppid_session_dataname"],
                    #      "SystemDateTime_BeginTrial": row["SystemDateTime_BeginTrial"]}
                )
        cprint(string=f"All items deleted from {table_name}.", col="g")

    else:
        cprint(string="Nothing will be  deleted.", col="g")


def _read_trial_results(process: bool = False, date: str | None = None) -> pd.DataFrame:
    """
    Get the full trial table of the main study.

    This table must have been downloaded as `csv` from `DynamoDB` (`"UXFData.FaceSim.TrialResults"`) before.

    :param process: True: force (re-)processing of data
    :param date: date of data to load (e.g., '2022-11-29')
    :return: processed trial table
    """
    # Check date
    if date is not None and not (date.startswith("202") and date.count("-") == 2):  # noqa: PLR2004
        msg = "'date' must be in format: 'YYYY-MM-DD'!"
        raise ValueError(msg)

    p2_trial_result_files = list(
        Path(paths.data.MAIN).glob(f"{'*' if date is None else date}_UXFData.FaceSim.TrialResults.csv")
    )

    if len(p2_trial_result_files) > 1:
        cprint(string="Found multiple trial result files. Choose one:", col="y", fm="ul")
        print("", *list(p2_trial_result_files), sep="\n\t")
        p2_trial_result_files = browse_files(initialdir=paths.data.MAIN, filetypes="*.csv")
        p2_trial_result_files = p2_trial_result_files.replace("_processed", "")
    else:
        p2_trial_result_files = p2_trial_result_files.pop()

    # Check if the full (i.e., concatenated) table is already there
    p2_raw_trial_result_files = (
        str(p2_trial_result_files).replace(paths.data.MAIN, paths.data.main.archive).replace(".csv", "_raw.csv")
    )

    if Path(p2_raw_trial_result_files).exists() and not process:
        # We know that p2_trial_result_files.is_file() is True
        table_processed = pd.read_csv(p2_trial_result_files)

    else:
        # Process the table before returning it
        table_processed = load_local_table(table_name=p2_trial_result_files.name)

        # Remove unnecessary columns & sort the rest
        table_processed = table_processed[SORTED_COLS]

        # Remove debug & UnlockTriplets users
        table_processed = table_processed.loc[
            table_processed.ppid.isin([p for p in table_processed.ppid if "debug" not in p])
        ]
        table_processed = table_processed[table_processed.ppid != "UnlockTriplets"].reset_index(drop=True)

        # Sort rows
        table_processed = table_processed.sort_values(
            by=["ppid", "SystemDateTime_BeginTrial"], axis=0, ascending=True, inplace=False
        ).reset_index(drop=True)

        # Archive unprocessed table
        p2_trial_result_files.rename(p2_raw_trial_result_files)

        # Save processed table
        table_processed.to_csv(p2_trial_result_files, index=False)

    return table_processed


def read_and_convert_s3_results_json_data(verbose: bool = False) -> pd.DataFrame:
    """
    Get the full trial table of the main study from memory.

    This table must be constructed from several `json` tables downloaded from `S3`.

    :param verbose: Be verbose or not
    :return: Processed trial table
    """
    # Define the session path
    p2_s3 = Path(paths.data.main.s3, "TrialResults")

    if verbose:
        tree(p2_s3)

    # Find the correct directory
    trial_result_dirs = [d for d in os.listdir(p2_s3) if (d.startswith("2022-") and not d.endswith(".csv"))]

    if len(trial_result_dirs) > 1:
        cprint(string="Choose random *.json* file from folder which should be unpacked:", col="y")
        trial_result_dirs = Path(browse_files(initialdir=p2_s3)).parent  # , filetypes=".json"))
    else:
        trial_result_dirs = trial_result_dirs.pop()
    trial_result_dirs = Path(p2_s3, trial_result_dirs)

    # Set path to processed trial table
    trial_result_full_table_path = Path(paths.data.MAIN, f"{trial_result_dirs.name}_UXFData.FaceSim.TrialResults.csv")

    # Check if the full (concatenated) table is already there
    convert_json_to_csv = True
    append_table = False
    table_processed = None  # one table to save all data
    if trial_result_full_table_path.is_file():
        table_processed = pd.read_csv(trial_result_full_table_path)
        cprint(string=f"\nTable with trial results already exists: {trial_result_full_table_path}", col="g")
        append_table = convert_json_to_csv = ask_true_false(
            question="Do you want to append new data to the existing table?", col="y"
        )

    if convert_json_to_csv:
        # Read json files
        trial_result_files = [f for f in os.listdir(trial_result_dirs) if (f.endswith((".json", ".json.gz")))]

        type_dict = None  # init DynamoDB type dict
        for json_file_path in tqdm(trial_result_files, desc="Read json files", position=0):
            p2_json = Path(trial_result_dirs, json_file_path)

            # Process json file via pandas
            table_raw = pd.read_json(p2_json, lines=True)

            if type_dict is None:
                td = [(col, next(iter(table_raw.iloc[0]["Item"][col].keys()))) for col in table_raw.iloc[0]["Item"]]

                td = pd.DataFrame(td, columns=["col_name", "type"])
                type_dict = dict(zip(td["col_name"], td["type"].map(DT_MAP), strict=True))

            # Remove type info from json file
            for row in tqdm(table_raw.values, desc=f"Read rows of '{json_file_path}'", position=1):
                current_row = row.item()

                if append_table:
                    ppid = current_row["ppid"]["S"]
                    sys_t = convert_date_time(date_time=current_row["SystemDateTime_BeginTrial"]["S"])
                    if sys_t in table_processed.loc[table_processed.ppid == ppid].SystemDateTime_BeginTrial.to_list():
                        continue

                trial_tab = pd.DataFrame(current_row)
                # append empty row to table
                copy_row = trial_tab.iloc[0:1].copy()
                copy_row[~pd.isna(copy_row)] = np.nan
                trial_tab = pd.concat([trial_tab, copy_row])

                # Write non-nan-value in empty row for each column
                for col in trial_tab.columns:
                    trial_tab.iloc[-1][col] = trial_tab[col].dropna().item()

                # Keep only filled row
                trial_tab = trial_tab.iloc[-1:]

                # Exclude empty rows
                if "triplet" not in trial_tab.columns:
                    continue
                if (
                    (trial_tab.head1 == trial_tab.head2).all()
                    and (trial_tab.head2 == trial_tab.head3).all()
                    and not trial_tab.head2.item()
                ) or (not trial_tab.triplet.item()):
                    # These rows are empty after the experiment was stopped early, usually after 3 missed
                    # catch trials
                    continue

                # Concatenate to big table
                table_processed = trial_tab if table_processed is None else pd.concat([table_processed, trial_tab])
                # , ignore_index=True)

        # Fill empty slots with nan
        table_processed = table_processed.replace("", np.nan)

        # Adapt dtypes
        table_processed = table_processed.astype(type_dict)

        # Remove unnecessary columns & sort the rest
        table_processed = table_processed[SORTED_COLS]

        # Solve date issue
        table_processed.SystemDateTime_BeginTrial = table_processed.SystemDateTime_BeginTrial.map(convert_date_time)

        # Sort rows table by ppid and start time/date of trial
        table_processed = table_processed.sort_values(
            by=["SystemDateTime_BeginTrial", "ppid"], axis=0, ascending=True
        ).reset_index(drop=True)

        # Save table
        table_processed.to_csv(trial_result_full_table_path, index=False)

    return table_processed


def where_to_find_trial_and_log_data(set_nr: str, update_table: bool = False) -> pd.DataFrame:
    """Get information about in which files trial results and log data can be found for a given Set number."""
    # Path to look-up table
    table_name: str = "Where_are_TrialResults_and_Logs.csv"
    p2_where_table = list(Path(paths.data.MAIN).glob(f"*{table_name}"))
    if len(p2_where_table) > 1:
        msg = f"More than one table found:\n{p2_where_table}"
        raise AssertionError(msg)

    if (len(p2_where_table) == 0) or update_table:
        if len(p2_where_table) == 0:
            cprint(string="Generating where-to-find table ...", col="b")

        # Init table
        where_table = pd.DataFrame(columns=["set_nr", "table_name", "type"])

        # Find all trial results and log files
        list_of_trial_result_tables = list(Path(paths.data.MAIN).glob("*UXFData.FaceSim.TrialResults.csv"))
        list_of_log_tables = list(Path(paths.data.MAIN).glob("*UXFData.FaceSim.SessionLog.csv"))
        list_of_tables = list_of_trial_result_tables + list_of_log_tables

        # Iterate through different sets
        set_files = sorted(Path(paths.data.main.prolific).glob("*Participants-Set*"))
        for p2_ppid_set in tqdm(
            set_files, desc="Find tables for each Set", total=len(set_files), position=0, colour="#51F1EE"
        ):
            # Get set number
            current_set_nr = p2_ppid_set.name.split("-Set")[-1].split("_")[0]

            # Get participants of current Set
            ppid_set = read_prolific_participant_data(set_nr=current_set_nr)["Participant id"].to_list()

            # Populate table
            for p2_table in tqdm(
                list_of_tables,
                desc=f"Iterate through all tables for Set{current_set_nr}",
                total=len(list_of_tables),
                position=1,
                leave=False,
                colour="#51A4F1",
            ):
                if "TrialResults" in p2_table.name:
                    tr_table = _read_trial_results(process=False, date=p2_table.name.split("_")[0])

                    # Extract those with matching set_nr
                    ppid_set_tr = [f"{p}_{set_infix(current_set_nr)}_trial_results" for p in ppid_set]

                    if tr_table.ppid_session_dataname.isin(ppid_set_tr).any():
                        where_table = where_table.append(
                            {"set_nr": current_set_nr, "table_name": p2_table.name, "type": "TrialResults"},
                            ignore_index=True,
                        )
                else:  # "SessionLog" in p2_table.name
                    log_table = load_local_table(table_name=p2_table.name)
                    ppid_set_log = [f"{p}_{set_infix(current_set_nr)}_log" for p in ppid_set]
                    if log_table.ppid_session_dataname.isin(ppid_set_log).any():
                        where_table = where_table.append(
                            {"set_nr": current_set_nr, "table_name": p2_table.name, "type": "SessionLog"},
                            ignore_index=True,
                        )

        # Sort table by set number
        where_table = where_table.sort_values(by=["set_nr", "type", "table_name"], axis=0, ascending=True).reset_index(
            drop=True
        )

        # Save (or overwrite) table
        where_table.to_csv(Path(paths.data.MAIN) / table_name, index=False)

        # Return table
        return where_table[where_table.set_nr == set_nr]

    # else use:  # len(p2_where_table) == 1:
    p2_where_table = p2_where_table.pop()
    where_table = pd.read_csv(p2_where_table, dtype=object)

    # Check if set_nr is in table (if not update table)
    if len(where_table[where_table.set_nr == set_nr]) == 0:
        cprint(string=f"Set{set_nr} is not in table {p2_where_table.name}. Updating table ...", col="y")
        where_table = where_to_find_trial_and_log_data(set_nr=set_nr, update_table=True)

    # Return table
    return where_table[where_table.set_nr == set_nr]


def remove_invalid_trials(trial_results_table: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove invalid trials from a given trial results table."""
    original_len = len(trial_results_table)

    # Remove training trials
    trial_results_table = trial_results_table[trial_results_table.block_num > 1]

    # Remove trials without data
    trial_results_table = trial_results_table[~trial_results_table.caught.isna()]
    trial_results_table = trial_results_table[trial_results_table.caught != ""]
    trial_results_table = trial_results_table.astype({"caught": bool})

    # Remove trials of blocks with catches
    for ppid_session_dataname, tr_table in tqdm(
        trial_results_table.groupby("ppid_session_dataname"),
        desc="Clean trial results table",
        total=len(trial_results_table.ppid_session_dataname.unique()),
    ):
        _ppid = tr_table.ppid.unique().item()  # == ppid_session_dataname.split("_")[0]

        # Remove all trials of participants with 3+ catches
        n_catches = tr_table.caught.sum()
        catch_threshold = 3
        if n_catches >= catch_threshold:
            if verbose:
                cprint(f"Participant {_ppid} has {n_catches} missed catch trials. Removing all trials ...", col="y")
            trial_results_table = trial_results_table[
                trial_results_table.ppid_session_dataname != ppid_session_dataname
            ]
            continue  # all blocks were removed, hence we can jump to the next participant session

        for b_idx, tr_block in tr_table.groupby("block_num"):
            if tr_block.caught.sum() > 0:
                if verbose:
                    cprint(
                        f"Participant {_ppid} has missed catch trials in block {int(b_idx)}. Removing block ...",
                        col="y",
                    )
                trial_results_table = trial_results_table[
                    ~(
                        (trial_results_table.ppid_session_dataname == ppid_session_dataname)
                        & (trial_results_table.block_num == b_idx)
                    )
                ]

    # Remove catch trials
    catch_head_trial = 0.0
    trial_results_table = trial_results_table[trial_results_table.catch_head == catch_head_trial]

    # Remove time-outs
    n_remove = (trial_results_table.head_odd == 0).sum()
    if verbose and n_remove > 0:
        cprint(string=f"{n_remove} time-out trials will be removed ...", col="y")
    trial_results_table = trial_results_table[trial_results_table.head_odd != 0]

    # TODO: Remove trials of unrealistic response times (< X sec):  # noqa: FIX002
    #  check: determine_threshold_for_minimal_response_time()  # noqa: ERA001
    #  trial_results_table[trial_results_table.response_time < params.MIN_RT_2D]  # params.MIN_RT_3D  # noqa: ERA001
    #  n_rt_outliers = (trial_results_table.response_time < params.MIN_RT_2D).sum()  # noqa: ERA001
    pass

    # TODO: Remove trials with monotonous choice behavior (> Y-times same side).  # noqa: FIX002
    #  This could also entail repeating patterns (e.g., left-right-left-right...)
    pass

    # TODO: Define other criteria (e.g., BQS, etc.)  # noqa: FIX002
    pass

    if verbose:
        n_removed = original_len - len(trial_results_table)
        cprint(
            f"\n{n_removed} of original {original_len} ({n_removed / original_len:.1%}) trials were removed ...",
            col="y",
            fm="bo",
        )

    return trial_results_table.reset_index(drop=True)


def read_trial_results_of_set(set_nr: str, clean_trials: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Read all trial results of a given Set number."""
    where_tab = where_to_find_trial_and_log_data(set_nr=set_nr, update_table=False)
    tr_table = None  # init
    for p2_tr_table in where_tab[where_tab.type == "TrialResults"].table_name:
        if verbose:
            print("Load:", p2_tr_table)
        if tr_table is None:
            tr_table = _read_trial_results(process=False, date=p2_tr_table.split("_")[0])

        else:
            if verbose:
                print("Append:", p2_tr_table)
            tr_table = tr_table.append(
                _read_trial_results(process=False, date=p2_tr_table.split("_")[0]), ignore_index=True
            )

    # Remove participants from table which are not part of given Set (set_nr)
    prolific_set_ppids = read_prolific_participant_data(set_nr=set_nr)["Participant id"]

    th_multi_sub_sample: int = 20
    if not set(tr_table.ppid).issubset(set(prolific_set_ppids)) and int(set_nr.split(".")[-1]) < th_multi_sub_sample:
        # Ignore Sets of multi-sub-sample
        if verbose:
            cprint(
                string=f"{len(set(tr_table.ppid) - set(prolific_set_ppids))} participant(s) are in the "
                f"trial results table, but are not part of Set{set_nr}! They will be dropped ...",
                col="y",
            )
        tr_table = tr_table[tr_table.ppid.isin(prolific_set_ppids)]  # drop (out-of-set_nr) ppids
        tr_table = tr_table.reset_index(drop=True)

    if clean_trials:
        tr_table = remove_invalid_trials(trial_results_table=tr_table, verbose=verbose)

    # Set column types
    if tr_table.caught.isna().any() or (tr_table.caught == "").any():
        return tr_table
    # Convert 'caught' column only to boolean, when there are no NaNs, i.e., missing trials
    # This is due to pd.Series([True, False, np.nan]).astype(bool) -> pd.Series([True, False, True]), ...
    # ..and to this: pd.Series([True, False, ""]).astype(bool) -> pd.Series([True, False, False])
    return tr_table.astype({"caught": bool})


def save_merged_tables_of_set(set_nr: str) -> None:
    """Merge all tables of a given type ("`TrialResults`", "`SessionLog`") in a given Set."""
    where_tab = where_to_find_trial_and_log_data(set_nr=set_nr, update_table=False)

    for table_type in ["TrialResults", "SessionLog"]:
        if table_type == "TrialResults":
            table_merged = read_trial_results_of_set(set_nr=set_nr, clean_trials=False, verbose=True)
        else:
            table_merged = read_logs_of_set(set_nr=set_nr)

        table_name = where_tab[where_tab.type == table_type].table_name.values[0]  # noqa: PD011
        prefix_date, suffix = table_name.split("_")
        if table_type == "TrialResults":
            table_merged = table_merged.drop_duplicates().reset_index(drop=True)
        else:
            table_merged = table_merged.drop_duplicates(subset=["ppid"]).reset_index(drop=True)
        prefix_date_m = prefix_date[:10] + "m"
        table_merged.to_csv(Path(paths.data.MAIN, f"{prefix_date_m}_{suffix}"), index=False)

        # Move other tables to "archive" folder
        for table_name in where_tab[where_tab.type == table_type].table_name:
            Path(paths.data.MAIN, table_name).rename(Path(paths.data.MAIN, "archive", table_name))

    # Remove table location file in where_to_find_trial_and_log_data()
    Path(paths.data.MAIN, "Where_are_TrialResults_and_Logs.csv").unlink()

    cprint(
        string="Tables are merged and saved, former tables are moved to 'archive' folder.\n"
        f"Consider renaming current tables with prefix '{prefix_date_m}_UXFData.FaceSim*.csv'.\n"
        "Then rerun where_to_find_trial_and_log_data()!",
        col="y",
    )


def read_trial_results_of_participant(ppid: str, clean_trials: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Read all trial results of a given participant."""
    set_nrs = get_participant_set_numbers(ppid=ppid)
    tr_table = None  # init
    for set_nr in set_nrs:
        if tr_table is None:
            tr_table = read_trial_results_of_set(set_nr=set_nr, clean_trials=clean_trials, verbose=verbose)
        else:
            tr_table = pd.concat(
                objs=[tr_table, read_trial_results_of_set(set_nr=set_nr, clean_trials=clean_trials, verbose=verbose)]
            )

    return tr_table[tr_table.ppid == ppid].reset_index(drop=True)


@lru_cache(maxsize=24)
def read_trial_results_of_session(
    session: str, clean_trials: bool = False, drop_subsamples: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """Read all trial results of a given session."""
    if session.upper() not in params.SESSIONS:
        msg = f"Session '{session}' not in {params.SESSIONS}!"
        raise ValueError(msg)

    set_nrs_of_session = [s for s in get_list_of_acquired_sets() if s.split(".")[0] == session[0]]
    set_nr_sub_sample = f"{session[0]}.20"
    if drop_subsamples:
        set_nrs_of_session.remove(set_nr_sub_sample)
    else:
        cprint(string=f"Multi-subsample Set-{set_nr_sub_sample} is included in the returned table!", col="r")

    tr_table = None  # init
    for set_nr in set_nrs_of_session:
        if tr_table is None:
            tr_table = read_trial_results_of_set(set_nr=set_nr, clean_trials=clean_trials, verbose=verbose)
        else:
            tr_table = pd.concat(
                [tr_table, read_trial_results_of_set(set_nr=set_nr, clean_trials=clean_trials, verbose=verbose)],
                ignore_index=True,
            )

    return tr_table


def read_logs_of_set(set_nr: str) -> pd.DataFrame:
    """Read all log tables of a given Set number."""
    where_tab = where_to_find_trial_and_log_data(set_nr=set_nr, update_table=False)
    log_table = None  # init
    for p2_log_table in where_tab[where_tab.type == "SessionLog"].table_name:
        print("Load:", p2_log_table)
        if log_table is None:
            log_table = load_local_table(table_name=p2_log_table)

        else:
            print("Append:", p2_log_table)
            log_table = log_table.append(load_local_table(table_name=p2_log_table))

    # Remove participants from table which are not part of given Set (set_nr)
    prolific_set_ppids = read_prolific_participant_data(set_nr=set_nr)["Participant id"]
    if not (log_table.ppid.isin(prolific_set_ppids)).all():
        cprint(
            string=f"{len(log_table[~log_table.ppid.isin(prolific_set_ppids)])} participant(s) are in the "
            f"log table, but are not part of Set{set_nr}! They will be dropped ...",
            col="y",
        )
        log_table = log_table[log_table.ppid.isin(prolific_set_ppids)]  # drop (out-of-set_nr) ppids
        log_table = log_table.reset_index(drop=True)

    return log_table


def plot_triplet_matrix(triplet_table: pd.DataFrame, n_faces: int) -> plt.Figure:
    """Plot matrix of triplets."""
    triplet_table["triplet"] = triplet_table.triplet.map(lambda x: x.split("_"))
    sampling_mat = np.zeros((n_faces, n_faces))
    for _i, triplet_row in tqdm(iterable=triplet_table.iterrows(), desc="Fill count matrix of triplets"):
        triplet = [int(f_id) for f_id in triplet_row.triplet]
        for comb in combinations(triplet, r=2):
            sampling_mat[comb[0] - 1, comb[1] - 1] += 1
            sampling_mat[comb[1] - 1, comb[0] - 1] += 1

    sampling_mat /= n_faces - 2
    np.fill_diagonal(sampling_mat, np.nan)

    # Plot the sampling matrix
    fig, ax = plt.subplots(num=f"{datetime.now().replace(microsecond=0)} | Sampled triplets", figsize=(10, 8))
    h = sns.heatmap(sampling_mat, cmap="YlOrBr", vmin=0, vmax=1, ax=ax)
    h.set(
        title=f"{datetime.now().replace(microsecond=0)} | "
        f"{len(triplet_table) / np.math.comb(n_faces, 3):.1%} Sampled triplets | "
        f"{np.nanmin(sampling_mat):.1%}-{np.nanmax(sampling_mat):.1%} (min-max)"
    )
    fig.tight_layout()
    plt.show()
    return fig


def get_current_state_of_triplets(session: str, pilot: bool = params.PILOT, plot: bool = False) -> pd.DataFrame:
    """Get the current state of triplets (e.g., which triplet is currently in the experiment)."""
    session = session.upper()
    if session not in params.SESSIONS:
        msg = f"Session '{session}' not in {params.SESSIONS}!"
        raise ValueError(msg)

    # Append table name
    table_name = "UXFData.FaceSim.TripletsIDB." + session
    if pilot:
        table_name += ".Pilot"

    # Load table from DynamoDB
    triplet_table = load_table_from_dynamodb(table_name=table_name, save=False, merge=False)

    # Get how many of each status
    n_complete = len(triplet_table[triplet_table.status == "G"])
    n_unseen = len(triplet_table[triplet_table.status == "U"])
    n_lock = len(triplet_table[triplet_table.status == "L"])
    n_total = len(triplet_table)  # == n_complete + n_lock + n_unseen == np.math.comb(n_faces, 3)

    # Print information
    cprint(string=f"Current state of {session} triplets:", fm="ul", ts=True)
    cprint(string=f"\t> {n_complete / n_total:.1%} triplets are completed", col="g")
    cprint(string=f"\t> {n_unseen / n_total:.1%} triplets are unseen", col="y")
    print(f"\t> {n_lock / n_total:.1%} triplets are locked")
    print(f"Data from {table_name} on DynamoDB")

    if plot:
        fig = plot_triplet_matrix(
            triplet_table=triplet_table[triplet_table.status == "G"],
            n_faces=params.pilot.v2.n_faces if pilot else params.main.n_faces,
        )

        fig.savefig(
            Path(paths.data.MAIN) / f"{datetime.now().date()}_sampled_{session}"
            f"{'-pilot' if pilot else ''}-triplets.png"
        )

    return triplet_table


def finalized_triplets(session: str) -> list[int]:
    """
    Provide an overview of the finalized triplets.

    For the given session, provide an overview of the finalized triplets. & return a list of remaining
    triplets.
    """
    n_all_triplets = np.math.comb(params.main.n_faces, 3)

    good_sess_tr_table = read_trial_results_of_session(session=session, clean_trials=True, verbose=False)

    sampled_unique_triplets = good_sess_tr_table.triplet_id.astype(int).unique()

    cprint(
        string=f"{len(sampled_unique_triplets) / n_all_triplets:.1%} of all triplets were sampled & approved "
        f"in session {session}.",
        col="g",
    )

    remaining_triplets = sorted(set(range(1, n_all_triplets + 1)) - set(sampled_unique_triplets))

    print(f"Number of remaining triplets: {len(remaining_triplets)}")

    return remaining_triplets


def _update_status_item(
    dynamodb_table,
    table_name: str,
    data_row,
    key_names,
    new_status: str = "U",
) -> None:
    _response = dynamodb_table.update_item(
        Key=dict(zip(key_names, [data_row[key] for key in key_names], strict=True)),
        UpdateExpression="SET #st = :s",
        ExpressionAttributeValues={":s": new_status},
        ExpressionAttributeNames={"#st": "status"},
    )
    # Note: 'status' is a reserved word in DynamoDB, so it needs to be escaped with a '#'
    successful_request: int = 200
    if _response["ResponseMetadata"]["HTTPStatusCode"] != successful_request:
        cprint(f"Error setting triplet status to '{new_status}' of {data_row['triplet_id']} in {table_name}.", col="r")


def update_triplet_table_on_dynamodb(
    session: str, set_finalised_triplets_to_g: bool = False, delete_done_triplets: bool = False
) -> None:
    """
    Update the triplet table on `DynamoDB`.

    :param session: '2D' OR '3D'
    :param set_finalised_triplets_to_g: Set finalized triplets to 'G' (if not already done)
    :param delete_done_triplets: Whether to delete triplets that are done
    :return: None
    """
    if not ask_true_false(f"\nWas the latest trial data downloaded for session '{session}'?"):
        cprint(string=f"Download the latest trial data of the '{session}' session first.", col="r")
        return

    open_triplets = finalized_triplets(session=session)

    table_name = "UXFData.FaceSim.TripletsIDB." + session.upper()

    # Connect to DynamoDB
    dynamodb = boto3.resource("dynamodb", region_name="eu-central-1")  # connect to DynamoDB
    db_table = dynamodb.Table(table_name)

    # Get key names
    key_schema = db_table.key_schema
    key_names = [k["AttributeName"] for k in key_schema]

    # Load current state of triplet table
    cprint(string=f"Loading current state of {session} triplet table ...", col="b")
    response = db_table.scan()
    data = response["Items"]
    while "LastEvaluatedKey" in response:
        response = db_table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        data.extend(response["Items"])

    df_current_state = pd.DataFrame(data)  # transform to pd.DataFrame

    # Unlock triplets
    locked: str = "L"  # locked symbol
    if (df_current_state.status == locked).any():
        cprint(
            string=f"\nUnlocking {(df_current_state.status == locked).sum()} previously locked triplets ...", col="b"
        )
        for row in tqdm(data, desc=f"Unlocking items in {table_name}", colour="#02B580"):
            if row["status"] != locked:
                continue
            _update_status_item(
                dynamodb_table=db_table, table_name=table_name, data_row=row, key_names=key_names, new_status="U"
            )  # set "L" to "U"
        cprint(string=f"All previously locked triplets are now unlocked in {table_name}.", col="g")

    # Set open triplets to "U"
    cprint(string=f"\nResetting {len(open_triplets)} open triplets to 'U' ...", col="b")
    for row in tqdm(data, desc=f"Reset open triplets items in {table_name}", colour="#E86A03"):
        if row["triplet_id"] not in open_triplets or row["status"] == "U":
            continue
        _update_status_item(
            dynamodb_table=db_table, table_name=table_name, data_row=row, key_names=key_names, new_status="U"
        )  # reset open triplets
    cprint(string=f"All open triplets are now ready to be sampled in {table_name}.", col="g")

    # Set finalized triplets to "G" (if not already done)
    if set_finalised_triplets_to_g:
        cprint(string="\nSetting finalised triplets to 'G' ...", col="b")
        for row in tqdm(data, desc=f"Reset open triplets items in {table_name}", colour="#8F29E8"):
            if row["triplet_id"] not in open_triplets and row["status"] != "G":
                _update_status_item(
                    dynamodb_table=db_table, table_name=table_name, data_row=row, key_names=key_names, new_status="G"
                )
        cprint(string=f"All finalised triplets are now set to 'G' in {table_name}.", col="g")

    # Delete triplets that are done
    if delete_done_triplets:
        cprint(string="\nDeleting triplets that are done ...", col="b")
        # TODO: Test this (should work but will interfere with other functions as for cost-estimation)  # noqa: FIX002
        msg = "This is not tested yet. Be careful!"
        raise NotImplementedError(msg)
        with db_table.batch_writer() as batch:
            for row in tqdm(data, desc=f"Deleting items in {table_name}"):
                if row["status"] != "G":
                    continue
                batch.delete_item(Key=dict(zip(key_names, [row[key] for key in key_names], strict=True)))
        cprint(string=f"All done triplets deleted from {table_name}.", col="g")


def finalized_triplets_multi_sub_sample() -> list[int]:
    """
    Provide an overview of finalized triplets.

    For the given session of the multi-sampled-sub-sample provide an overview & return the list of remaining triplets.
    """
    n_all_triplets = np.math.comb(params.multisubsample.n_faces, 3)

    # The following includes the trials written in 'UXFData.FaceSim.OtherTrialData' as well
    trial_results_table = load_table_from_dynamodb(table_name="UXFData.FaceSim.TrialResults", save=False, merge=False)

    good_sess_tr_table = remove_invalid_trials(trial_results_table=trial_results_table, verbose=True)

    # Filter for multi-sub-sample
    sampled_triplets_counts = good_sess_tr_table.triplet_id.astype(int).value_counts()

    print("\n", sampled_triplets_counts, "\n")

    for i in range(1, params.multisubsample.n_reps + 1):
        cprint(
            string=f"{(sampled_triplets_counts >= i).sum()}/{n_all_triplets} "
            f"({(sampled_triplets_counts >= i).sum() / n_all_triplets:.1%}) of all triplets were sampled at "
            f"least {i} times!",
            col="g",
        )
    # perc_sampled_n_times = len(sampled_triplets_counts[  # noqa: ERA001, RUF100
    #         sampled_triplets_counts == params.multisubsample.n_reps]) / n_all_triplets

    # cprint(f"{perc_sampled_n_times:.1%} of all triplets were sampled {params.multisubsample.n_reps} times!", "g")  # noqa: ERA001, E501

    remaining_triplets = sorted(
        set(range(1, n_all_triplets + 1))
        - set(sampled_triplets_counts[sampled_triplets_counts == params.multisubsample.n_reps].index)
    )

    print(f"Number of remaining triplet IDs: {len(remaining_triplets)}")

    # done_triplets = sorted(set(  # noqa: ERA001, RUF100
    #     sampled_triplets_counts[sampled_triplets_counts == params.multisubsample.n_reps].index))

    # print(f"Number of done triplets: {len(done_triplets)}")  # noqa: ERA001

    return remaining_triplets


def update_triplet_table_on_dynamodb_multi_sub_sample(session: str, set_finalised_triplets_to_g: bool = True) -> None:
    """
    Update triplet table on `DynamoDB` for the given session of the `multi-sampled-sub-sample`.

    Note: This asserts that all data on `DynamoDB` is only from the given session.
    Do not execute this function, if also data from other sessions is on `DynamoDB`.
    """
    cprint(
        f"\nUpdating triplet table on DynamoDB for the {session}-session of the multi-sampled-sub-sample ...\n",
        col="y",
        fm="bo",
    )
    if not ask_true_false(
        question=f"Are you sure you want to update the triplet table for the {session}-session on "
        f"DynamoDB AND that only data of that session is currently on DynamoDB "
        f"(this is asserted)? "
    ):
        cprint(string="Aborting ...", col="r")
        return

    open_triplets = finalized_triplets_multi_sub_sample()

    table_name = "UXFData.FaceSim.TripletsIDB." + session.upper()

    # Connect to DynamoDB
    dynamodb = boto3.resource("dynamodb", region_name="eu-central-1")  # connect to DynamoDB
    db_table = dynamodb.Table(table_name)

    # Get key names
    key_schema = db_table.key_schema
    key_names = [k["AttributeName"] for k in key_schema]

    # Load current state of triplet table
    cprint(string=f"Loading current state of {session} triplet table ...", col="b")
    response = db_table.scan()
    data = response["Items"]
    while "LastEvaluatedKey" in response:
        response = db_table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        data.extend(response["Items"])

    df_current_state = pd.DataFrame(data)  # transform to pd.DataFrame

    # Unlock triplets
    locked: str = "L"  # locked symbol
    if (df_current_state.status == locked).any():
        cprint(
            string=f"\nUnlocking {(df_current_state.status == locked).sum()} previously locked triplets ...", col="b"
        )
        for row in tqdm(data, desc=f"Unlocking items in {table_name}"):
            if row["status"] != locked:
                continue
            _update_status_item(
                dynamodb_table=db_table, table_name=table_name, data_row=row, key_names=key_names, new_status="U"
            )  # set "L" to "U"
        cprint(string=f"All previously locked triplets are now unlocked in {table_name}.", col="g")

    # Set open triplets to "U"
    cprint(string=f"\nResetting {len(open_triplets)} open triplets to 'U' ...", col="b")
    for row in tqdm(data, desc=f"Reset open triplets items in {table_name}"):
        if row["triplet_id"] not in open_triplets or row["status"] == "U":
            continue
        _update_status_item(
            dynamodb_table=db_table, table_name=table_name, data_row=row, key_names=key_names, new_status="U"
        )  # reset open triplets
    cprint(string=f"All open triplets are now ready to be sampled in {table_name}.", col="g")

    # Set finalized triplets to "G" (if not already done)
    found_finalised_triplets = False
    if set_finalised_triplets_to_g:
        cprint(string="\nSetting finalised triplets to 'G' ...", col="b")
        for row in tqdm(data, desc=f"Reset open triplets items in {table_name}"):
            if row["triplet_id"] not in open_triplets and row["status"] != "G":
                _update_status_item(
                    dynamodb_table=db_table, table_name=table_name, data_row=row, key_names=key_names, new_status="G"
                )
                found_finalised_triplets = True
        if found_finalised_triplets:
            cprint(string=f"All finalised triplets are now set to 'G' in {table_name}.", col="g")

    if len(open_triplets) == 0:
        cprint(string=f"\nAll triplets are finalised in {table_name}.", col="g")
    else:
        cprint(
            string=f"\nInvite max {np.maximum(np.floor(len(open_triplets) / 171).astype(int), 1)} "
            f"participants at once!\n",
            col="y",
            fm="bo",
        )


def main() -> None:
    """Run the main function of `read_data.py`."""
    table_list = [
        "UXFData.FaceSim." + name
        for name in [
            "OtherSessionData",
            "ParticipantDetails",
            "Settings",
            "SessionLog",
            "TrialResults",
            "OtherTrialData",
        ]
    ]  # "OtherTrialData" must be last & after "TrialResults"

    if FLAGS.mss:
        # Set all other boolean FLAGS to False
        for flag in FLAGS.__dict__:
            if flag not in {"mss", "set_nr", "triplets"}:
                setattr(FLAGS, flag, False)
            # FLAGS.load = FLAGS.delete FLAGS.plot = FLAGS.pilot = FLAGS.verbose = False

        update_triplet_table_on_dynamodb_multi_sub_sample(
            session=f"{FLAGS.set_nr[0]}D", set_finalised_triplets_to_g=True
        )

    if FLAGS.load:
        for table_name in table_list[:-1]:  # "*.OtherTrialData" will be merged with "*.TrialResults"
            tab = load_table_from_dynamodb(table_name=table_name, save=True, merge=True)
            if FLAGS.verbose:
                print(tab)

    # Delete all items in all tables (but *TripletsIDB.*D)
    if FLAGS.delete:
        for table_name in table_list:
            delete_all_items_in_table_on_dynamodb(table_name=table_name)

    if FLAGS.verbose:
        cprint(string=f"\nLoad data of Set{FLAGS.set_nr} ...", col="b")
        tab = read_trial_results_of_set(set_nr=FLAGS.set_nr, clean_trials=False, verbose=True)
        print(tab)

    if FLAGS.triplets:
        get_current_state_of_triplets(session=FLAGS.triplets, pilot=FLAGS.pilot, plot=FLAGS.plot)


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Add arg parser
    parser = argparse.ArgumentParser(description="Read data gathered via DynamoDB")  # init arg parser

    parser.add_argument(
        "-s",
        "--set_nr",
        type=str,
        action="store",
        default=SET_NUMBER,
        help="Define Set of 2D session: (2.0, 2.1, ...) OR 3D session: (3.0, 3.1, ...)",
    )

    parser.add_argument(
        "-l",
        "--load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load all data from DynamoDB (takes a while)",
    )

    parser.add_argument(
        "-d", "--delete", action=argparse.BooleanOptionalAction, default=False, help="Delete all data from DynamoDB"
    )

    parser.add_argument(
        "-t",
        "--triplets",
        type=str,
        default=None,
        help="Get info of session ('2D' or '3D') about current state of triplets table on DynamoDB",
    )

    parser.add_argument(
        "-p", "--plot", action=argparse.BooleanOptionalAction, default=True, help="Plot sampled triplets."
    )

    parser.add_argument(
        "--mss", action=argparse.BooleanOptionalAction, default=False, help="Process multi-sub-sample data"
    )

    parser.add_argument("--pilot", action=argparse.BooleanOptionalAction, default=False, help="Use pilot data")

    parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, default=False, help="Verbose")

    # Parse arguments
    FLAGS, unparsed = parser.parse_known_args()

    # %% Run main
    main()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
