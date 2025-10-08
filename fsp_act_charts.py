# fsp_act_charts.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Data loading & preparation
# -----------------------------
def load_and_clean(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalise key categoricals if present
    for col in ("activity", "source", "mine", "region"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def make_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Return weekly aggregates for delay, cycles, and mean cycle duration."""
    delay = (
        df.loc[df["activity"] == "track_mtx_delay"]
          .groupby(["week", "source"], as_index=False)["duration"]
          .sum()
          .rename(columns={"duration": "total_mtx_delay_min"})
    )

    cycles = (
        df.groupby(["week", "source"], as_index=False)["completed_cycles"]
          .max()
          .rename(columns={"completed_cycles": "completed_cycles_week"})
    )

    mean_cycle_df = (
        df.loc[df["activity"] != "track_mtx_delay"]
            .groupby(['mine', 'week', 'source'], as_index=False)
            .agg(duration_sum=('duration','sum'),
                 total_cycles = ("completed_cycles", 'max'))
    )

    mean_cyc = (
        mean_cycle_df.groupby(["week", "source"], as_index=False)
        .agg({'duration_sum': 'sum', 'total_cycles': 'sum'})
        .assign(mean_cycle_duration_min=lambda x:x['duration_sum']/x['total_cycles'])
        # .drop(columns=['duration_sum', 'total_cycles'])
    )

    # mean_cyc = (
    #     mean_cycle_df.groupby(["week", "source"], as_index=False)["mean_cycle_time"]
    #       .mean()
    #       .rename(columns={"cycle_time": "mean_cycle_duration"})
    # )

    weekly = (mean_cyc
              .merge(delay, on=["week", "source"], how="left")
              # .merge(mean_cyc, on=["week", "source"], how="left"))
              )
    weekly["total_mtx_delay_min"] = weekly["total_mtx_delay_min"].fillna(0.0)
    return weekly


def label_delay_level(weekly: pd.DataFrame, high_quantile: float = 0.75) -> pd.DataFrame:
    """Label each week as high/low by total maintenance delay across all sources."""
    delay_per_week = (
        weekly.groupby("week", as_index=False)["total_mtx_delay_min"]
              .sum()
              .rename(columns={"total_mtx_delay_min": "delay_all_sources_min"})
    )
    threshold = float(delay_per_week["delay_all_sources_min"].quantile(high_quantile))
    delay_per_week["delay_level"] = np.where(
        delay_per_week["delay_all_sources_min"] >= threshold, "high", "low"
    )
    return weekly.merge(delay_per_week[["week", "delay_all_sources_min", "delay_level"]],
                        on="week", how="left")


def prep_reg_df(weekly_labeled: pd.DataFrame) -> pd.DataFrame:
    """Minimised frame for charts."""
    reg = weekly_labeled.copy()
    reg["delay_hours"] = reg["total_mtx_delay_min"] / 60.0
    return reg


# -----------------------------
# Charts (matplotlib)
# -----------------------------
def chart_scatter_delay_vs_cycles(reg_df: pd.DataFrame) -> plt.Figure:
    """Scatter: maintenance delay (hours) vs completed cycles, coloured by source."""
    fig = plt.figure()
    for src in sorted(reg_df["source"].dropna().unique()):
        s = reg_df.loc[reg_df["source"] == src]
        plt.scatter(s["delay_hours"], s["completed_cycles_week"], label=src.upper(), alpha=0.75)
    plt.xlabel("Track maintenance delay (hours)")
    plt.ylabel("Completed cycles per week")
    plt.title("Delay vs Completed Cycles (FSP vs ACT)")
    plt.legend()
    plt.tight_layout()
    return fig


def chart_box_cycles_by_bucket_source(reg_df: pd.DataFrame) -> plt.Figure:
    """Boxplots: completed cycles grouped by (delay_level × source)."""
    df = reg_df.dropna(subset=["completed_cycles_week", "delay_level", "source"]).copy()
    df["group"] = df["delay_level"].str.title() + " delay - " + df["source"].str.upper()

    order = ["High delay - ACT", "High delay - FSP", "Low delay - ACT", "Low delay - FSP"]
    # keep only existing groups, in preferred order
    order = [g for g in order if g in df["group"].unique()]

    data = [df.loc[df["group"] == g, "completed_cycles_week"].to_numpy() for g in order]

    fig = plt.figure()
    plt.boxplot(data, labels=order, showmeans=True)
    plt.ylabel("Completed cycles per week")
    plt.title("Completed Cycles by Delay Bucket and Source")
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def chart_box_duration_by_bucket_source(weekly_labeled: pd.DataFrame) -> plt.Figure:
    """Boxplots: mean cycle duration (minutes) grouped by (delay_level × source)."""
    df = weekly_labeled.dropna(
        subset=["mean_cycle_duration_min", "delay_level", "source"]
    ).copy()
    df["group"] = df["delay_level"].str.title() + " delay - " + df["source"].str.upper()

    order = ["High delay - ACT", "High delay - FSP", "Low delay - ACT", "Low delay - FSP"]
    order = [g for g in order if g in df["group"].unique()]

    data = [df.loc[df["group"] == g, "mean_cycle_duration_min"].to_numpy() for g in order]

    fig = plt.figure()
    plt.boxplot(data, labels=order, showmeans=True)
    plt.ylabel("Mean cycle duration (minutes)")
    plt.title("Mean Cycle Duration by Delay Bucket and Source")
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


# -----------------------------
# Runner
# -----------------------------
def run(path: str | Path, high_quantile: float = 0.75, out_dir: str | Path = "./charts") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = load_and_clean(path)
    weekly = make_weekly(df)
    weekly_labeled = label_delay_level(weekly, high_quantile=high_quantile)
    reg_df = prep_reg_df(weekly_labeled)

    # Create and save plots
    chart_scatter_delay_vs_cycles(reg_df).savefig(Path(out_dir) / "scatter_delay_vs_cycles.png", dpi=300)
    chart_box_cycles_by_bucket_source(reg_df).savefig(Path(out_dir) / "box_cycles_by_bucket.png", dpi=300)
    chart_box_duration_by_bucket_source(weekly_labeled).savefig(Path(out_dir) / "box_duration_by_bucket.png", dpi=300)

    print(f"Charts saved to: {Path(out_dir).resolve()}")



if __name__ == "__main__":
    # Update to your CSV path
    CSV_PATH = r"C:\Users\61432\Downloads\combined_df.csv"
    OUT_PATH = r"C:\Users\61432\Downloads"
    run(CSV_PATH, high_quantile=0.75, out_dir=OUT_PATH)
