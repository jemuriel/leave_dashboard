from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Tuple


# -----------------------------
# I/O & Normalisation
# -----------------------------
def load_roster(path: str | Path, *, dt_format: str = "%m/%d/%Y %H:%M") -> pd.DataFrame:
    """
    Load the roster (leave) dataset and normalise key columns.
    Expects columns: NAME, DATE, STATE, TASK (others preserved).
    """
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"], format=dt_format, errors="coerce")
    df["STATE"] = df["STATE"].astype(str).str.strip().str.upper()
    df["DATE_ONLY"] = df["DATE"].dt.date
    return df


def load_flag_table(
    path: str | Path,
    *,
    kind: Literal["public_holiday", "shutdown"],
    encoding: str = "cp1252",
) -> pd.DataFrame:
    """
    Load a flag table with columns STATE, DATE (one row per day), and return:
    [STATE, DATE_ONLY, IS_PUBLIC_HOL] or [STATE, DATE_ONLY, IS_SHUTDOWN]
    """
    df = pd.read_csv(path, encoding=encoding)
    if "STATE" not in df.columns or "DATE" not in df.columns:
        raise ValueError(f"{kind} file must contain columns: STATE, DATE")

    df = df.copy()
    df["STATE"] = df["STATE"].astype(str).str.strip().str.upper()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["DATE_ONLY"] = df["DATE"].dt.date
    df = df[["STATE", "DATE_ONLY"]].dropna().drop_duplicates()

    flag_col = "IS_PUBLIC_HOL" if kind == "public_holiday" else "IS_SHUTDOWN"
    df[flag_col] = True
    return df


def merge_flags(base: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge flag table into the base roster by (STATE, DATE_ONLY).
    Any missing flags become False.
    """
    out = base.merge(flags, on=["STATE", "DATE_ONLY"], how="left")
    for col in ("IS_PUBLIC_HOL", "IS_SHUTDOWN"):
        if col in out.columns:
            out[col] = out[col].fillna(False)
    return out


# -----------------------------
# Segments & Statistics
# -----------------------------
def label_segments(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute segment ids per NAME: seg_id increments at each TASK == 'WORK'.
    Return (df_with_seg, valid_between_mask).
    """
    df = df.sort_values(["NAME", "DATE"], kind="mergesort").reset_index(drop=True)
    df["seg_id"] = df.groupby("NAME")["TASK"].transform(lambda s: s.eq("WORK").cumsum())

    between_rows = df["TASK"].ne("WORK")
    last_seg = df.groupby("NAME")["seg_id"].transform("max")
    valid_between = between_rows & (df["seg_id"] < last_seg)

    return df, valid_between


def compute_segment_stats(df: pd.DataFrame, valid_between: pd.Series) -> pd.DataFrame:
    """
    Compute per-(NAME, seg_id) stats across rows strictly between WORKs.
    Includes counts for AN, PL, DIL, and intersections with public holidays & shutdowns.
    """
    is_AN = df["TASK"].eq("AN")
    is_PL = df["TASK"].eq("PL")
    is_DIL = df["TASK"].eq("DIL")  # NEW: DIL flag

    # Segment length
    seg_len = (
        df.loc[valid_between]
          .groupby(["NAME", "seg_id"], sort=False)
          .size()
          .rename("SEG_LEN")
    )

    # Totals by leave type
    total_AN = (
        df.loc[valid_between & is_AN]
          .groupby(["NAME", "seg_id"], sort=False)
          .size()
          .rename("TOTAL_AN")
    )
    total_PL = (
        df.loc[valid_between & is_PL]
          .groupby(["NAME", "seg_id"], sort=False)
          .size()
          .rename("TOTAL_PL")
    )
    total_DIL = (  # NEW: total DIL rows in each segment
        df.loc[valid_between & is_DIL]
          .groupby(["NAME", "seg_id"], sort=False)
          .size()
          .rename("TOTAL_DIL")
    )

    # Intersections with Public Holidays
    an_hol = (
        df.loc[valid_between & is_AN & df.get("IS_PUBLIC_HOL", False)]
          .groupby(["NAME", "seg_id"], sort=False)
          .size()
          .rename("AN_HOL_COUNT")
    )
    pl_hol = (
        df.loc[valid_between & is_PL & df.get("IS_PUBLIC_HOL", False)]
          .groupby(["NAME", "seg_id"], sort=False)
          .size()
          .rename("PL_HOL_COUNT")
    )
    dil_hol = (  # NEW
        df.loc[valid_between & is_DIL & df.get("IS_PUBLIC_HOL", False)]
          .groupby(["NAME", "seg_id"], sort=False)
          .size()
          .rename("DIL_HOL_COUNT")
    )

    # Intersections with Shutdowns
    an_shut = (
        df.loc[valid_between & is_AN & df.get("IS_SHUTDOWN", False)]
          .groupby(["NAME", "seg_id"], sort=False)
          .size()
          .rename("AN_SHUT_COUNT")
    )
    pl_shut = (
        df.loc[valid_between & is_PL & df.get("IS_SHUTDOWN", False)]
          .groupby(["NAME", "seg_id"], sort=False)
          .size()
          .rename("PL_SHUT_COUNT")
    )
    dil_shut = (  # NEW
        df.loc[valid_between & is_DIL & df.get("IS_SHUTDOWN", False)]
          .groupby(["NAME", "seg_id"], sort=False)
          .size()
          .rename("DIL_SHUT_COUNT")
    )

    seg_stats = pd.concat(
        [
            seg_len,
            total_AN, total_PL, total_DIL,  # NEW: include DIL totals
            an_hol, pl_hol, dil_hol,        # NEW: include DIL holiday intersections
            an_shut, pl_shut, dil_shut      # NEW: include DIL shutdown intersections
        ],
        axis=1,
    ).fillna(
        {
            "TOTAL_AN": 0,
            "TOTAL_PL": 0,
            "TOTAL_DIL": 0,       # NEW
            "AN_HOL_COUNT": 0,
            "PL_HOL_COUNT": 0,
            "DIL_HOL_COUNT": 0,   # NEW
            "AN_SHUT_COUNT": 0,
            "PL_SHUT_COUNT": 0,
            "DIL_SHUT_COUNT": 0,  # NEW
        }
    )

    return seg_stats


def build_output_columns(seg_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-(NAME, seg_id) output columns.
    Mirrors AN/PL behaviour and adds DIL the same way as PL.
    """
    out = pd.DataFrame(index=seg_stats.index)
    out["COUNT_BETWEEN_WORK"] = seg_stats["SEG_LEN"].astype(int)
    out["COUNT_BETWEEN_WORK_IF_ANY_AN"] = np.where(
        seg_stats["TOTAL_AN"] > 0, seg_stats["SEG_LEN"], 0
    ).astype(int)

    # Totals by leave type
    out["TOTAL_NUMBER_OF_AN_BETWEEN_WORK"] = seg_stats["TOTAL_AN"].astype(int)
    out["TOTAL_NUMBER_OF_PL_BETWEEN_WORK"] = seg_stats["TOTAL_PL"].astype(int)
    out["TOTAL_NUMBER_OF_DIL_BETWEEN_WORK"] = seg_stats["TOTAL_DIL"].astype(int)  # NEW

    # Booleans (any intersection) — PL logic replicated for DIL
    out["AN_INTERSECTS_PUBLIC_HOL"] = seg_stats["AN_HOL_COUNT"].astype(int).gt(0)
    out["PL_INTERSECTS_PUBLIC_HOL"] = seg_stats["PL_HOL_COUNT"].astype(int).gt(0)
    out["DIL_INTERSECTS_PUBLIC_HOL"] = seg_stats["DIL_HOL_COUNT"].astype(int).gt(0)  # NEW

    out["AN_INTERSECTS_SHUTDOWN"] = seg_stats["AN_SHUT_COUNT"].astype(int).gt(0)
    out["PL_INTERSECTS_SHUTDOWN"] = seg_stats["PL_SHUT_COUNT"].astype(int).gt(0)
    out["DIL_INTERSECTS_SHUTDOWN"] = seg_stats["DIL_SHUT_COUNT"].astype(int).gt(0)  # NEW

    return out


def annotate_first_rows(df: pd.DataFrame, valid_between: pd.Series, out_cols: pd.DataFrame) -> pd.DataFrame:
    """
    Write out_cols values only on the first row (earliest DATE) of each (NAME, seg_id)
    among rows that are strictly between two WORKs. Drop seg_id afterwards.
    """
    first_idx = (
        df.loc[valid_between]
          .groupby(["NAME", "seg_id"], sort=False)["DATE"]
          .idxmin()
    )

    for col in out_cols.columns:
        df[col] = np.nan

    first_keys = df.loc[first_idx, ["NAME", "seg_id"]]
    first_values = first_keys.join(out_cols, on=["NAME", "seg_id"])
    df.loc[first_idx, out_cols.columns] = first_values[out_cols.columns].to_numpy()

    return df.drop(columns=["seg_id"])


# -----------------------------
# Orchestration
# -----------------------------
def process_leave_segments(
    roster_path: str | Path,
    holidays_path: str | Path,
    shutdowns_path: str | Path,
    *,
    dt_format: str = "%m/%d/%Y %H:%M",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    End-to-end pipeline:
      1) loads roster,
      2) merges public holiday & shutdown flags,
      3) computes segment stats (AN, PL, DIL),
      4) annotates first rows per segment with the results,
      5) optionally saves to CSV.

    Returns the final DataFrame.
    """
    df = load_roster(roster_path, dt_format=dt_format)
    hol = load_flag_table(holidays_path, kind="public_holiday")
    shut = load_flag_table(shutdowns_path, kind="shutdown")

    df = merge_flags(df, hol)
    df = merge_flags(df, shut)

    df_seg, valid_between = label_segments(df)
    seg_stats = compute_segment_stats(df_seg, valid_between)
    out_cols = build_output_columns(seg_stats)
    df_final = annotate_first_rows(df_seg, valid_between, out_cols)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)

    return df_final


# -----------------------------
# Script entry point
# -----------------------------
def main():
    roster_path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\new_data_analysis\Leave_new_data.csv"
    holidays_path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\new_data_analysis\public_holidays.csv"
    shutdowns_path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\new_data_analysis\shutdowns.csv"
    output_path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\new_data_analysis\leave_analysis_new_data_v3.csv"

    process_leave_segments(
        roster_path=roster_path,
        holidays_path=holidays_path,
        shutdowns_path=shutdowns_path,
        dt_format="%m/%d/%Y %H:%M",
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
