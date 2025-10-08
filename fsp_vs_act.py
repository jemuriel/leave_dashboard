# streamlit_app.py
import math

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from typing import List

st.set_page_config(page_title="Activity per Cycle Dashboard", layout="wide")

DEFAULT_PATH = r"C:\Users\61432\Downloads\combined_df.csv"


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(path: str | None, uploaded) -> pd.DataFrame:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif path:
        df = pd.read_csv(path)
    else:
        raise ValueError("No data source provided.")
    # Standardise column names (trim + lower)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


LEGEND_LAYOUT = dict(
    legend_title_text="Activity",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


def check_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Available columns: {list(df.columns)}")


@st.cache_data
def compute_ratio_of_sums(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    g = (
        df.groupby(dims, dropna=False, as_index=False)
        .agg(duration_sum=("duration", "sum"),
             completed_sum=("completed_cycles", "sum"))
    )
    g["value"] = np.where(g["completed_sum"].eq(0), np.nan,
                          g["duration_sum"] / g["completed_sum"])
    return g


def order_weeks(df: pd.DataFrame, week_col: str = "week") -> List:
    """
    Produce a sensible ordering for 'week'. Works if week is numeric, or a string.
    If weeks look like '2025-W09' or '2025-09', tries to sort naturally; otherwise lexical.
    """
    weeks = df[week_col].dropna().unique().tolist()
    # Try numeric ordering
    try:
        return sorted(weeks, key=lambda x: float(x))
    except Exception:
        pass

    # Try to normalise strings like '2025-W09'
    def _week_key(w):
        s = str(w)
        # Pull year & number tokens if present
        tokens = []
        for t in s.replace("W", "-").replace("_", "-").replace("/", "-").split("-"):
            try:
                tokens.append(int(t))
            except Exception:
                tokens.append(t)
        return tokens

    return sorted(weeks, key=_week_key)


# -----------------------------
# Define global activity colour mapping
# -----------------------------
def make_activity_colors(df: pd.DataFrame) -> dict:
    """Assign a consistent colour per activity."""
    activities = sorted(df["activity"].dropna().unique())
    palette = px.colors.qualitative.Set1  # you can change to Set1, Dark2, etc.
    # repeat palette if more activities than colours
    colors = (palette * ((len(activities) // len(palette)) + 1))[: len(activities)]
    return dict(zip(activities, colors))


# -----------------------------
# Replace make_stacked_chart
# -----------------------------
import pandas as pd
import plotly.express as px


# Pass the exact bottom-to-top order you want in `activity_order`.
# The first item will be the BOTTOM of the stack; the last will be the TOP.

def _stacked_max(df: pd.DataFrame, x_col: str = "week", y_col: str = "value") -> float:
    """Max stacked height across x (sum of y over categories at each x)."""
    if df.empty:
        return 0.0
    # If x_col is categorical, observed=True avoids unused categories
    s = df.groupby(x_col, observed=True)[y_col].sum()
    return float(s.max()) if len(s) else 0.0


def _nice_ceiling(x: float) -> float:
    """Round x up to a 'nice' tick boundary."""
    if x <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(x))
    for mult in (1, 2, 2.5, 5, 10):
        if mult * mag >= x:
            return mult * mag
    return 10 * mag


def common_y_range_for_pair(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        x_col: str = "week",
        y_col: str = "value",
        pad: float = 0.05,  # 5% headroom
        nice: bool = True,  # round to nice tick
) -> tuple[float, float]:
    """Return (ymin, ymax) that fits both stacked charts built from df_a and df_b."""
    m = max(_stacked_max(df_a, x_col, y_col), _stacked_max(df_b, x_col, y_col))
    cap = 0.0 if m == 0 else m * (1 + pad)
    if nice:
        cap = _nice_ceiling(cap)
    return (0.0, cap)


# ---------- your chart builder, now with optional y_range control ----------
def make_stacked_chart(
        df_one_source: pd.DataFrame,
        source_name: str,
        week_order: list,
        showlegend: bool = True,
        activity_order: list | None = None,  # bottom->top stack order
        reverse_legend: bool = False,
        y_range: tuple[float, float] | None = None,  # <<< set this with common_y_range_for_pair
) -> px.bar:
    if activity_order is None:
        # deterministic first-seen order
        activity_order = list(dict.fromkeys(df_one_source["activity"].tolist()))

    fig = px.bar(
        df_one_source,
        x="week",
        y="value",
        color="activity",
        category_orders={
            "week": week_order,
            "activity": activity_order,
        },
        color_discrete_map=activity_colors,  # assumes pre-defined dict
        barmode="stack",
        title=f"{source_name}",
        labels={
            "week": "Week",
            "value": "Duration / Completed cycles",
            "activity": "Activity",
        },
    )

    if y_range is not None:
        fig.update_yaxes(range=y_range)

    fig.update_layout(
        xaxis_type="category",
        title=dict(font=dict(size=14)),
        margin=dict(l=10, r=10, t=30, b=60),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.25,
            xanchor="left", x=0,
            font=dict(size=10),
            title_text="Activity",
            traceorder="reversed" if reverse_legend else "normal",
        ),
    )
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("Stacked Columns — Mean Duration per Cycle (ratio of sums)")
st.caption(
    "For each (source, activity, week): value = sum(duration) / sum(completed_cycles). "
    "Top: one chart per source. Bottom: a matrix of charts by Region (rows) × Source (columns)."
)

# Load + validate
df = pd.read_csv(DEFAULT_PATH)

# Build mapping once
activity_colors = make_activity_colors(df)

# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar:
    st.header("Filters")
    all_activities = sorted(df["activity"].dropna().unique())
    activities_sel = st.multiselect(
        "Activity",
        options=all_activities,
        default=all_activities,
        help="Select one or more activities to include across all charts.",
    )

# Apply filter to a working copy used by both sections
# --- Week filter (affects ALL charts) ---
all_weeks = order_weeks(df)  # keeps natural ordering for weeks
week_sel = st.multiselect(
    "Weeks",
    options=all_weeks,
    default=all_weeks,
    help="Select one or more weeks to include across all charts.",
)

# Apply filters to the working copy used by all sections
df_f = df.copy()
if activities_sel:
    df_f = df_f[df_f["activity"].isin(activities_sel)]
else:
    st.warning("No activities selected. Select at least one activity to display charts.")
    st.stop()

# ← NEW: apply week filter to the same working frame used everywhere
if week_sel:
    df_f = df_f[df_f["week"].isin(week_sel)]
else:
    st.warning("No weeks selected. Select at least one week to display charts.")
    st.stop()


# 2) Build BOTH aggregates
ratio_df = compute_ratio_of_sums(df_f, ["source", "week", "activity"])  # top charts
ratio_df_reg = compute_ratio_of_sums(df_f, ["region", "source", "week", "activity"])  # region x source grid

if ratio_df.empty:
    st.warning("No aggregated rows to display (check your filters/data).")
    st.stop()

# week_order = order_weeks(ratio_df, "week")
week_order = order_weeks(pd.concat([ratio_df[["week"]], ratio_df_reg[["week"]]]))

activity_order = ['Travel (empty) to mine',
                  'Load at mine',
                  'Travel (loaded) to port',
                  'Dump at port',
                  'Travel (empty) to yard',
                  'track_mtx_delay',
                  ]

# -----------------------------
# Section 1: By Source — side-by-side using columns
# -----------------------------
st.header("By Source")

sources = sorted(ratio_df["source"].dropna().unique())

if not sources:
    st.info("No 'source' values found to render the charts.")
else:
    # st.caption("Each chart is a Source; legends shown on every chart.")
    charts_per_row = 4


    def _chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]


    y_range = (0, ratio_df.groupby(['week', 'source'])['value'].sum().max() * 1.05)

    for row_sources in _chunks(sources, charts_per_row):
        cols = st.columns(len(row_sources))
        for j, src in enumerate(row_sources):
            with cols[j]:
                subset = ratio_df[ratio_df["source"] == src]
                if subset.empty:
                    st.caption(f"_{src}: no data_")
                    continue
                fig = make_stacked_chart(subset, f"{src}", week_order, showlegend=True, y_range=y_range,
                                         activity_order=activity_order)
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Section 2: Grid by Region (rows) × Source (columns) — side-by-side using columns
# -----------------------------
st.header("By Region × Source")

regions = sorted(ratio_df_reg["region"].dropna().unique())
all_sources = sorted(ratio_df_reg["source"].dropna().unique())

if not regions or not all_sources:
    st.info("No 'region' or 'source' values found to render the matrix.")
else:
    st.caption("Each row is a Region; columns are Sources. Legends shown on every chart.")
    y_range = (0, ratio_df_reg.groupby(['week', 'source', 'region'])['value'].sum().max() * 1.05)
    for reg in regions:
        st.markdown(f"**Region: {reg}**")
        row_df = ratio_df_reg[ratio_df_reg["region"] == reg]

        cols = st.columns(len(all_sources))
        for j, src in enumerate(all_sources):
            with cols[j]:
                subset = row_df[row_df["source"] == src]
                if subset.empty:
                    st.caption(f"_{src}: no data_")
                    continue
                fig = make_stacked_chart(subset, f"{src}", week_order, showlegend=True, y_range=y_range,
                                         activity_order=activity_order)
                st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Section 3: FSP vs ACT Summary Charts (Plotly)
# -----------------------------
import plotly.express as px
import numpy as np
import pandas as pd

st.header("FSP vs ACT Summary (Plotly)")
st.caption(
    "How track maintenance (‘track_mtx_delay’) relates to completed cycles and mean cycle durations. "
    "Weeks are bucketed into High/Low delay based on the weekly total across all sources (75th percentile)."
)

# --- Helpers: mirror fsp_act_charts.py but Plotly-friendly ---
def _make_weekly_plotly(df_raw: pd.DataFrame) -> pd.DataFrame:
    # normalise key categoricals if present (safe even if already lower)
    df = df_raw.copy()
    for col in ("activity", "source", "mine", "region", "week"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # total maintenance delay minutes per (week, source)
    delay = (
        df.loc[df["activity"] == "track_mtx_delay"]
          .groupby(["week", "source"], as_index=False)["duration"]
          .sum()
          .rename(columns={"duration": "total_mtx_delay_min"})
    )

    # completed cycles per (week, source): take max across rows in the week
    # cycles = (
    #     df.groupby(["week", "source"], as_index=False)["completed_cycles"]
    #       .max()
    #       .rename(columns={"completed_cycles": "completed_cycles_week"})
    # )

    # mean cycle duration = sum(duration for non-delay activities) / sum(cycles) per (week, source)
    # first aggregate by (mine, week, source) to avoid double counting
    mean_cycle_df = (
        df.loc[df["activity"] != "track_mtx_delay"]
          .groupby(["mine", "week", "source"], as_index=False)
          .agg(duration_sum=("duration", "sum"),
               total_cycles=("completed_cycles", "max"))
    )

    mean_cyc = (
        mean_cycle_df.groupby(["week", "source"], as_index=False)
          .agg(duration_sum=("duration_sum", "sum"),
               total_cycles=("total_cycles", "sum"))
          .assign(mean_cycle_duration_min=lambda x: np.where(x["total_cycles"].eq(0), np.nan,
                                                             x["duration_sum"] / x["total_cycles"]))
          .drop(columns=["duration_sum"])
    )

    weekly = (mean_cyc
              .merge(delay, on=["week", "source"], how="left")
              # .merge(mean_cyc, on=["week", "source"], how="left"))
              )
    weekly["total_mtx_delay_min"] = weekly["total_mtx_delay_min"].fillna(0.0)
    return weekly

def _label_delay_level_plotly(weekly: pd.DataFrame, high_quantile: float = 0.75) -> pd.DataFrame:
    delay_per_week = (
        weekly.groupby("week", as_index=False)["total_mtx_delay_min"]
              .sum()
              .rename(columns={"total_mtx_delay_min": "delay_all_sources_min"})
    )
    threshold = float(delay_per_week["delay_all_sources_min"].quantile(high_quantile))
    delay_per_week["delay_level"] = np.where(
        delay_per_week["delay_all_sources_min"] >= threshold, "high", "low"
    )
    return weekly.merge(
        delay_per_week[["week", "delay_all_sources_min", "delay_level"]], on="week", how="left"
    )

def _prep_reg_df_plotly(weekly_labeled: pd.DataFrame) -> pd.DataFrame:
    reg = weekly_labeled.copy()
    # reg["delay_hours"] = reg["total_mtx_delay_min"] / 60.0
    # Keep upper-cased SOURCE labels for prettier legends
    reg["SOURCE_LABEL"] = reg["source"].str.upper()
    # Also a tidy group label for the box plots
    reg["group"] = reg["delay_level"].str.title() + " delay - " + reg["SOURCE_LABEL"]
    return reg

# --- Build weekly + buckets from the same dataset already loaded as `df` ---
weekly = _make_weekly_plotly(df_f)  # uses the df loaded earlier in the app
weekly_labeled = _label_delay_level_plotly(weekly, high_quantile=0.75)
reg_df = _prep_reg_df_plotly(weekly_labeled)

# -----------------------------
# Chart 1: Scatter — Delay (hours) vs Completed Cycles
# -----------------------------
fig_scatter = px.scatter(
    reg_df,
    x="total_mtx_delay_min",
    y="total_cycles",
    color="SOURCE_LABEL",
    hover_data={"week": True, "SOURCE_LABEL": True, "total_mtx_delay_min": ":.1f", "total_cycles": True},
    labels={
        "delay_hours": "Track maintenance delay (minutes)",
        "total_cycles": "Completed cycles per week",
        "SOURCE_LABEL": "Source",
    },
    title="Delay vs Completed Cycles (FSP vs ACT)",
)
fig_scatter.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)
st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------
# Toggle: X-axis grouping mode
# -----------------------------
grouping_mode = st.radio(
    "X-axis grouping",
    ["Source (ACT vs FSP)", "Delay bucket × Source (High/Low × ACT/FSP)"],
    horizontal=True,
)

# Build x grouping for reg_df (used by the first two boxplots)
if grouping_mode.startswith("Source"):
    # source-only: ACT vs FSP
    reg_df["x_group"] = reg_df["SOURCE_LABEL"]  # 'ACT' / 'FSP'
    order = [g for g in ["ACT", "FSP"] if g in reg_df["x_group"].unique()]
    x_tickangle = 0
else:
    # High/Low × Source buckets
    reg_df["x_group"] = reg_df["group"]  # 'High delay - ACT', etc.
    wanted = ["High delay - ACT", "High delay - FSP", "Low delay - ACT", "Low delay - FSP"]
    order = [g for g in wanted if g in reg_df["x_group"].unique()]
    x_tickangle = 0

# -----------------------------
# Chart 2 (updated): Box — Completed Cycles by chosen grouping
# -----------------------------
fig_box_cycles = px.box(
    reg_df.dropna(subset=["total_cycles", "x_group"]),
    x="x_group",
    y="total_cycles",
    category_orders={"x_group": order},
    points=False,
    labels={"x_group": "", "total_cycles": "Completed cycles per week"},
    title="Completed Cycles by Grouping",
)
fig_box_cycles.update_layout(
    xaxis_tickangle=x_tickangle,
    showlegend=False,
    margin=dict(l=10, r=10, t=40, b=40),
)
st.plotly_chart(fig_box_cycles, use_container_width=True)

# -----------------------------
# Chart 3 (updated): Box — Mean Cycle Duration (minutes) by chosen grouping
# -----------------------------
fig_box_duration = px.box(
    reg_df.dropna(subset=["mean_cycle_duration_min", "x_group"]),
    x="x_group",
    y="mean_cycle_duration_min",
    category_orders={"x_group": order},
    points=False,
    labels={"x_group": "", "mean_cycle_duration_min": "Mean cycle duration (minutes)"},
    title="Mean Cycle Duration by Grouping",
)
fig_box_duration.update_layout(
    xaxis_tickangle=x_tickangle,
    showlegend=False,
    margin=dict(l=10, r=10, t=40, b=40),
)
st.plotly_chart(fig_box_duration, use_container_width=True)

# -----------------------------
# Extra Chart (updated): Mean Delay per Cycle (Boxplot) by chosen grouping
# -----------------------------
delay_cycle_df = weekly_labeled.copy()
delay_cycle_df["mean_delay_per_cycle_min"] = np.where(
    delay_cycle_df["total_cycles"].eq(0),
    np.nan,
    delay_cycle_df["total_mtx_delay_min"] / delay_cycle_df["total_cycles"],
)

# prep labels to mirror reg_df
delay_cycle_df["SOURCE_LABEL"] = delay_cycle_df["source"].str.upper()
delay_cycle_df["group"] = delay_cycle_df["delay_level"].str.title() + " delay - " + delay_cycle_df["SOURCE_LABEL"]

if grouping_mode.startswith("Source"):
    delay_cycle_df["x_group"] = delay_cycle_df["SOURCE_LABEL"]
    order_delay = [g for g in ["ACT", "FSP"] if g in delay_cycle_df["x_group"].unique()]
else:
    delay_cycle_df["x_group"] = delay_cycle_df["group"]
    wanted_delay = ["High delay - ACT", "High delay - FSP", "Low delay - ACT", "Low delay - FSP"]
    order_delay = [g for g in wanted_delay if g in delay_cycle_df["x_group"].unique()]

fig_box_mean_delay = px.box(
    delay_cycle_df.dropna(subset=["mean_delay_per_cycle_min", "x_group"]),
    x="x_group",
    y="mean_delay_per_cycle_min",
    category_orders={"x_group": order_delay},
    labels={"x_group": "", "mean_delay_per_cycle_min": "Mean delay per cycle (minutes)"},
    points=False,
    title="Mean Track Maintenance Delay per Cycle by Grouping",
)
fig_box_mean_delay.update_layout(
    xaxis_tickangle=x_tickangle,
    showlegend=False,
    margin=dict(l=10, r=10, t=40, b=40),
)
st.plotly_chart(fig_box_mean_delay, use_container_width=True)

