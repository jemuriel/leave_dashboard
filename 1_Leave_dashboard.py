# streamlit_app.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Leave Granularity", layout="wide")

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_FOLDER = PROJECT_ROOT / "csv"

DEFAULT_PL_CSV = CSV_FOLDER / "an_pl_ph_data.csv"
SENTINEL_YEARS = 125  # special value

# NEW (DIL heatmap): default path to uploaded data
DEFAULT_DIL_CSV = CSV_FOLDER / "an_dil_data.csv"

# =============================================================================
# HELPERS (DATA LOADING / CLEANING)
# =============================================================================
@st.cache_data
def load_csv_upper(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def coerce_blocksize(df: pd.DataFrame) -> pd.DataFrame:
    if "BLOCK_SIZE" not in df.columns:
        raise ValueError("Missing required column 'BLOCK_SIZE'")
    out = df.copy()
    out["BLOCK_SIZE"] = pd.to_numeric(out["BLOCK_SIZE"], errors="coerce")
    out = out.dropna(subset=["BLOCK_SIZE"]).copy()
    out["BLOCK_SIZE"] = out["BLOCK_SIZE"].astype(int)
    return out


def coerce_if_any_pl01(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "IF_ANY_PL" not in out.columns:
        return out
    s = out["IF_ANY_PL"]
    out["IF_ANY_PL"] = (
        pd.to_numeric(s, errors="coerce").gt(0)
        .fillna(
            s.astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "1.0", "y", "yes", "true", "t", "pl"})
        )
        .astype(int)
    )
    return out


def coerce_if_any_st01(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of coerce_if_any_pl01 for IF_ANY_ST (if present)."""
    out = df.copy()
    if "IF_ANY_ST" not in out.columns:
        return out
    s = out["IF_ANY_ST"]
    out["IF_ANY_ST"] = (
        pd.to_numeric(s, errors="coerce").gt(0)
        .fillna(
            s.astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "1.0", "y", "yes", "true", "t", "st"})
        )
        .astype(int)
    )
    return out


# =============================================================================
# HELPERS (HEATMAP DATA & CHARTS)
# =============================================================================
def make_heatmap(pivot: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    # Keep only rows (y categories) that have at least one non-zero cell across x
    numeric = pivot.apply(pd.to_numeric, errors="coerce").fillna(0)
    row_has_values = numeric.gt(0).any(axis=1)

    # If nothing has values, fall back to the original (avoids empty-figure issues)
    data = pivot.loc[row_has_values] if row_has_values.any() else pivot

    x_labels = [str(x) for x in data.columns]
    y_labels = [str(y) for y in data.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=data.values,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            colorbar=dict(title="Employees"),
            hovertemplate="Block size: %{x}<br>%{y}<br>Employees: %{z}<extra></extra>",
            zmin=0,
        )
    )
    fig.update_xaxes(
        title="Block size (days)",
        type="category",
        categoryorder="array",
        categoryarray=x_labels,
        tickmode="array",
        tickvals=x_labels,
        ticktext=x_labels,
        dtick=1,
    )
    fig.update_yaxes(
        autorange="reversed",
        title=y_title,
        type="category",
        categoryorder="array",
        categoryarray=y_labels,
        dtick=1,
    )
    fig.update_layout(title=title, margin=dict(l=60, r=40, t=70, b=50))
    return fig


def pivot_allblocks(df_blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks_per_person = (
        df_blocks.groupby("NAME", as_index=False)
        .size()
        .rename(columns={"size": "NUM_BLOCKS"})
    )
    df_uniq = (
        df_blocks.drop_duplicates(["NAME", "BLOCK_SIZE"])
        .merge(blocks_per_person, on="NAME", how="left")
    )

    df_uniq["BLOCK_SIZE_BUCKET"] = df_uniq["BLOCK_SIZE"].where(
        df_uniq["BLOCK_SIZE"] <= 30, ">30"
    )
    df_uniq["NUM_BLOCKS_BUCKET"] = df_uniq["NUM_BLOCKS"].where(
        df_uniq["NUM_BLOCKS"] <= 10, ">10"
    )

    x_order = list(range(1, 31)) + [">30"]
    y_order = list(range(1, 11)) + [">10"]

    heat = (
        df_uniq.groupby(["NUM_BLOCKS_BUCKET", "BLOCK_SIZE_BUCKET"])
        .size()
        .reset_index(name="COUNT")
    )
    pivot = (
        heat.pivot(index="NUM_BLOCKS_BUCKET", columns="BLOCK_SIZE_BUCKET", values="COUNT")
        .reindex(index=y_order, columns=x_order)
        .fillna(0)
    )
    return pivot, df_uniq


def pivot_plonly(df_pl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks_per_person_pl = (
        df_pl.groupby("NAME", as_index=False)
        .size()
        .rename(columns={"size": "NUM_PL_BLOCKS"})
    )
    name_block_pl = df_pl.drop_duplicates(["NAME", "BLOCK_SIZE"])
    df_uniq_pl = name_block_pl.merge(blocks_per_person_pl, on="NAME", how="left")

    df_uniq_pl["BLOCK_SIZE_BUCKET"] = df_uniq_pl["BLOCK_SIZE"].where(
        df_uniq_pl["BLOCK_SIZE"] <= 30, ">30"
    )
    df_uniq_pl["NUM_BLOCKS_BUCKET"] = df_uniq_pl["NUM_PL_BLOCKS"].where(
        df_uniq_pl["NUM_PL_BLOCKS"] <= 10, ">10"
    )

    x_order = list(range(1, 31)) + [">30"]
    y_order = list(range(1, 11)) + [">10"]

    heat_pl = (
        df_uniq_pl.groupby(["NUM_BLOCKS_BUCKET", "BLOCK_SIZE_BUCKET"])
        .size()
        .reset_index(name="COUNT")
    )
    pivot_pl = (
        heat_pl.pivot(index="NUM_BLOCKS_BUCKET", columns="BLOCK_SIZE_BUCKET", values="COUNT")
        .reindex(index=y_order, columns=x_order)
        .fillna(0)
    )
    return pivot_pl, df_uniq_pl


# ---------- NEW (DIL heatmap) ----------
@st.cache_data
def load_dil_csv(path: str | Path) -> pd.DataFrame:
    """Load the DIL file (expects at least DIL_BLOCK_SIZE and BLOCK_SIZE)."""
    df = pd.read_csv(path)
    return df


def pivot_dil_blocks(df_dil: pd.DataFrame) -> pd.DataFrame:
    """
    Build a heatmap table with:
      - X: BLOCK_SIZE (1..30, >30)
      - Y: DIL_BLOCK_SIZE (1..30, >30)
      - Z: count of rows
    We coerce both sizes to numeric integers and bucket like the first heatmap's x-axis.
    """
    required = {"BLOCK_SIZE", "DIL_BLOCK_SIZE"}
    if not required.issubset(df_dil.columns):
        raise ValueError(f"DIL CSV must contain columns: {required}")

    d = df_dil.copy()
    d["BLOCK_SIZE"] = pd.to_numeric(d["BLOCK_SIZE"], errors="coerce")
    d["DIL_BLOCK_SIZE"] = pd.to_numeric(d["DIL_BLOCK_SIZE"], errors="coerce")
    d = d.dropna(subset=["BLOCK_SIZE", "DIL_BLOCK_SIZE"])
    if d.empty:
        return pd.DataFrame()

    d["BLOCK_SIZE"] = d["BLOCK_SIZE"].astype(int)
    d["DIL_BLOCK_SIZE"] = d["DIL_BLOCK_SIZE"].astype(int)

    # Bucket exactly as elsewhere
    d["X_BUCKET"] = d["BLOCK_SIZE"].where(d["BLOCK_SIZE"] <= 30, ">30")
    d["Y_BUCKET"] = d["DIL_BLOCK_SIZE"].where(d["DIL_BLOCK_SIZE"] <= 30, ">30")

    x_order = list(range(1, 31)) + [">30"]
    y_order = list(range(1, 31)) + [">30"]

    heat = (
        d.groupby(["Y_BUCKET", "X_BUCKET"])
        .size()
        .reset_index(name="COUNT")
    )
    pivot = (
        heat.pivot(index="Y_BUCKET", columns="X_BUCKET", values="COUNT")
        .reindex(index=y_order, columns=x_order)
        .fillna(0)
    )
    return pivot
# --------------------------------------


def make_pl_rate_chart(df: pd.DataFrame, bars_as_percent: bool = True) -> go.Figure:
    if not {"BLOCK_SIZE", "IF_ANY_PL"}.issubset(df.columns):
        return go.Figure()

    size = pd.to_numeric(df["BLOCK_SIZE"], errors="coerce")
    d = df.loc[size.notna() & (size > 0)].copy()
    size = size.loc[d.index]
    d["SIZE_BUCKET"] = np.where(
        size > 15, ">15", size.clip(upper=15).astype(int).astype(str)
    )

    ct = (
        d.groupby(["SIZE_BUCKET", "IF_ANY_PL"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "no_pl", 1: "pl"})
    )

    if ct.empty:
        return go.Figure()

    ct["n"] = ct["no_pl"] + ct["pl"]
    total_n = ct["n"].sum()
    ct["pct_total"] = (ct["n"] / total_n * 100).fillna(0)
    ct["pl_rate"] = (ct["pl"] / ct["n"] * 100).fillna(0)

    cats = [str(i) for i in range(1, 15 + 1)] + [">15"]
    ct = ct.reindex(cats)

    if bars_as_percent:
        y_bars = ct["pct_total"].fillna(0)
        bar_name = "% of total blocks"
        bar_hover = "Block size %{x}<br>% of total %{y:.1f}%<extra></extra>"
        left_title = "% of total blocks"
    else:
        y_bars = ct["n"].fillna(0)
        bar_name = "# blocks"
        bar_hover = "Block size %{x}<br>Count %{y}<extra></extra>"
        left_title = "# blocks"

    y_line = ct["pl_rate"].fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=ct.index.tolist(),
        y=y_bars.tolist(),
        name=bar_name,
        hovertemplate=bar_hover,
    )
    fig.add_scatter(
        x=ct.index.tolist(),
        y=y_line.tolist(),
        name="% with PL",
        mode="lines+markers",
        hovertemplate="Block size %{x}<br>PL rate %{y:.1f}%<extra></extra>",
        secondary_y=True,
    )

    left_max = float(y_bars.max()) if len(y_bars) else 1.0
    right_max = float(y_line.max()) if len(y_line) else 1.0
    left_max = left_max or 1.0
    right_max = right_max or 1.0

    fig.update_xaxes(
        title_text="Block size (days)",
        type="category",
        categoryorder="array",
        categoryarray=cats,
        tickmode="array",
        tickvals=cats,
        ticktext=cats,
        tickangle=0
    )
    fig.update_layout(xaxis=dict(automargin=True))

    fig.update_yaxes(title_text=left_title, range=[0, left_max], secondary_y=False)
    fig.update_yaxes(title_text="% with PL", range=[0, right_max], secondary_y=True)
    fig.update_layout(
        title="PL rate vs Block Size (bucketed)",
        legend_title="",
        bargap=0.15,
        margin=dict(l=60, r=40, t=70, b=50),
    )
    return fig

def make_pl_rate_chart_anph_true(df: pd.DataFrame, bars_as_percent: bool = True) -> go.Figure:
    """
    Same as make_pl_rate_chart, but only for rows where AN_PH == True.
    """
    required = {"BLOCK_SIZE", "IF_ANY_PL", "AN_PH"}
    if not required.issubset(df.columns):
        return go.Figure()

    col = df["AN_PH"]
    if pd.api.types.is_bool_dtype(col):
        mask_true = col
    else:
        mask_true = col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})

    d0 = df.loc[mask_true == True].copy()
    if d0.empty:
        return go.Figure()

    size = pd.to_numeric(d0["BLOCK_SIZE"], errors="coerce")
    d = d0.loc[size.notna() & (size > 0)].copy()
    if d.empty:
        return go.Figure()

    size = size.loc[d.index]
    d["SIZE_BUCKET"] = np.where(size > 15, ">15", size.clip(upper=15).astype(int).astype(str))

    # Ensure IF_ANY_PL is 0/1 for grouping
    if set(pd.unique(d["IF_ANY_PL"].dropna())) - {0, 1}:
        s = pd.to_numeric(d["IF_ANY_PL"], errors="coerce").fillna(0)
        d["IF_ANY_PL"] = (s > 0).astype(int)

    # st.write(d)
    ct = d.groupby(["SIZE_BUCKET", "IF_ANY_PL"]).size().unstack()
    for col_id in (0, 1):
        if col_id not in ct.columns:
            ct[col_id] = 0
    ct = ct[[0, 1]].rename(columns={0: "no_pl", 1: "pl"})

    if ct.empty:
        return go.Figure()
    ct = ct.fillna(0)
    ct["n"] = ct["no_pl"] + ct["pl"]
    total_n = ct["n"].sum()
    ct["pct_total"] = (ct["n"] / total_n * 100).fillna(0)
    ct["pl_rate"] = (ct["pl"] / ct["n"] * 100).fillna(0)

    cats = [str(i) for i in range(1, 16)] + [">15"]
    ct = ct.reindex(cats)
    # st.write(ct)

    if bars_as_percent:
        y_bars = ct["pct_total"].fillna(0)
        bar_name = "% of PH blocks"
        bar_hover = "Block size %{x}<br>% of PH total %{y:.1f}%<extra></extra>"
        left_title = "% of PH blocks"
    else:
        y_bars = ct["n"].fillna(0)
        bar_name = "# PH blocks"
        bar_hover = "Block size %{x}<br>Count %{y}<extra></extra>"
        left_title = "# PH blocks"

    y_line = ct["pl_rate"].fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=ct.index.tolist(),
        y=y_bars.tolist(),
        name=bar_name,
        hovertemplate=bar_hover,
    )
    fig.add_scatter(
        x=ct.index.tolist(),
        y=y_line.tolist(),
        name="% with PL (PH only)",
        mode="lines+markers",
        hovertemplate="Block size %{x}<br>PL rate (PH only) %{y:.1f}%<extra></extra>",
        secondary_y=True,
    )

    left_max = float(y_bars.max()) if len(y_bars) else 1.0
    right_max = float(y_line.max()) if len(y_line) else 1.0
    left_max = left_max or 1.0
    right_max = right_max or 1.0

    fig.update_xaxes(
        title_text="Block size (days)",
        type="category",
        categoryorder="array",
        categoryarray=cats,
        tickmode="array",
        tickvals=cats,
        ticktext=cats,
        tickangle=0
    )
    fig.update_layout(xaxis=dict(automargin=True))

    fig.update_yaxes(title_text=left_title, range=[0, left_max], secondary_y=False)
    fig.update_yaxes(title_text="% with PL (PH only)", range=[0, right_max], secondary_y=True)
    fig.update_layout(
        title="PL rate vs Block Size — Public holidays only",
        legend_title="",
        bargap=0.15,
        margin=dict(l=60, r=40, t=70, b=50),
    )
    return fig


def make_pl_rate_chart_anst_true(df: pd.DataFrame, bars_as_percent: bool = True) -> go.Figure:
    """
    Same as make_pl_rate_chart, but only for rows where AN_ST == True (Shutdowns).
    """
    required = {"BLOCK_SIZE", "IF_ANY_PL", "AN_ST"}
    if not required.issubset(df.columns):
        return go.Figure()

    col = df["AN_ST"]
    if pd.api.types.is_bool_dtype(col):
        mask_true = col
    else:
        mask_true = col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})

    d0 = df.loc[mask_true == True].copy()
    if d0.empty:
        return go.Figure()

    size = pd.to_numeric(d0["BLOCK_SIZE"], errors="coerce")
    d = d0.loc[size.notna() & (size > 0)].copy()
    if d.empty:
        return go.Figure()

    size = size.loc[d.index]
    d["SIZE_BUCKET"] = np.where(size > 15, ">15", size.clip(upper=15).astype(int).astype(str))

    # Ensure IF_ANY_PL is 0/1
    if set(pd.unique(d["IF_ANY_PL"].dropna())) - {0, 1}:
        s = pd.to_numeric(d["IF_ANY_PL"], errors="coerce").fillna(0)
        d["IF_ANY_PL"] = (s > 0).astype(int)

    ct = d.groupby(["SIZE_BUCKET", "IF_ANY_PL"]).size().unstack()
    for col_id in (0, 1):
        if col_id not in ct.columns:
            ct[col_id] = 0
    ct = ct[[0, 1]].rename(columns={0: "no_pl", 1: "pl"})

    if ct.empty:
        return go.Figure()
    ct = ct.fillna(0)
    ct["n"] = ct["no_pl"] + ct["pl"]
    total_n = ct["n"].sum()
    ct["pct_total"] = (ct["n"] / total_n * 100).fillna(0)
    ct["pl_rate"] = (ct["pl"] / ct["n"] * 100).fillna(0)

    cats = [str(i) for i in range(1, 16)] + [">15"]
    ct = ct.reindex(cats)

    if bars_as_percent:
        y_bars = ct["pct_total"].fillna(0)
        bar_name = "% of ST blocks"
        bar_hover = "Block size %{x}<br>% of ST total %{y:.1f}%<extra></extra>"
        left_title = "% of ST blocks"
    else:
        y_bars = ct["n"].fillna(0)
        bar_name = "# ST blocks"
        bar_hover = "Block size %{x}<br>Count %{y}<extra></extra>"
        left_title = "# ST blocks"

    y_line = ct["pl_rate"].fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=ct.index.tolist(),
        y=y_bars.tolist(),
        name=bar_name,
        hovertemplate=bar_hover,
    )
    fig.add_scatter(
        x=ct.index.tolist(),
        y=y_line.tolist(),
        name="% with PL (ST only)",
        mode="lines+markers",
        hovertemplate="Block size %{x}<br>PL rate (ST only) %{y:.1f}%<extra></extra>",
        secondary_y=True,
    )

    left_max = float(y_bars.max()) if len(y_bars) else 1.0
    right_max = float(y_line.max()) if len(y_line) else 1.0
    left_max = left_max or 1.0
    right_max = right_max or 1.0

    fig.update_xaxes(
        title_text="Block size (days)",
        type="category",
        categoryorder="array",
        categoryarray=cats,
        tickmode="array",
        tickvals=cats,
        ticktext=cats,
        tickangle=0
    )
    fig.update_layout(xaxis=dict(automargin=True))

    fig.update_yaxes(title_text=left_title, range=[0, left_max], secondary_y=False)
    fig.update_yaxes(title_text="% with PL (ST only)", range=[0, right_max], secondary_y=True)
    fig.update_layout(
        title="PL rate vs Block Size — Shutdowns only",
        legend_title="",
        bargap=0.15,
        margin=dict(l=60, r=40, t=70, b=50),
    )
    return fig


def sidebar_filter_controls(df: pd.DataFrame):
    # (unchanged content)
    # --- keep your existing implementation here ---
    with st.sidebar:
        st.header("Filters")
        sel_depo_groups = None
        if "DEPO_GROUP" in df.columns:
            depo_group_opts = sorted(df["DEPO_GROUP"].dropna().unique().tolist())
            sel_depo_groups = st.multiselect(
                "DEPO_GROUP",
                options=depo_group_opts,
                default=st.session_state.get("sel_depo_groups", depo_group_opts)
                if depo_group_opts
                else None,
                key="sel_depo_groups",
            )

        base_for_depo = df.copy()
        if sel_depo_groups is not None and "DEPO_GROUP" in base_for_depo.columns:
            base_for_depo = base_for_depo[
                base_for_depo["DEPO_GROUP"].isin(sel_depo_groups)
            ]

        sel_depos = None
        if "DEPO" in base_for_depo.columns:
            depo_opts = sorted(base_for_depo["DEPO"].dropna().unique().tolist())
            prev_sel_depos = st.session_state.get("sel_depos", depo_opts)
            preserved = [d for d in prev_sel_depos if d in depo_opts]
            sel_depos = st.multiselect(
                "DEPO",
                options=depo_opts,
                default=(preserved if preserved else depo_opts) if depo_opts else None,
                key="sel_depos",
            )

        _base = df.copy()
        if sel_depo_groups is not None and "DEPO_GROUP" in _base.columns:
            _base = _base[_base["DEPO_GROUP"].isin(sel_depo_groups)]
        if sel_depos is not None and "DEPO" in _base.columns:
            _base = _base[_base["DEPO"].isin(sel_depos)]

        sel_gender = None
        if "GENDER" in _base.columns:
            gender_opts = sorted(_base["GENDER"].dropna().unique().tolist())
            prev_sel_gender = st.session_state.get("sel_gender", gender_opts)
            sel_gender = st.multiselect(
                "GENDER",
                options=gender_opts,
                default=[g for g in prev_sel_gender if g in gender_opts]
                if gender_opts
                else None,
                key="sel_gender",
            )

        sel_job = None
        if "JOB_TYPE" in _base.columns:
            job_opts = sorted(_base["JOB_TYPE"].dropna().unique().tolist())
            prev_sel_job = st.session_state.get("sel_job", job_opts)
            sel_job = st.multiselect(
                "JOB_TYPE",
                options=job_opts,
                default=[g for g in prev_sel_job if g in job_opts]
                if job_opts
                else None,
                key="sel_job",
            )

        sel_years_range = None
        include_125 = False
        if "YEARS_OF_WORK" in _base.columns:
            years_numeric = pd.to_numeric(_base["YEARS_OF_WORK"], errors="coerce").dropna()
            years_for_slider = years_numeric[years_numeric != SENTINEL_YEARS]

            if not years_for_slider.empty:
                y_min = float(years_for_slider.min())
                y_max = float(years_for_slider.max())
                step_is_int = (years_for_slider % 1 == 0).all()
                step = 1 if step_is_int else 0.1

                prev_lo, prev_hi = st.session_state.get("sel_years_range", (y_min, y_max))
                default_lo = max(y_min, float(prev_lo))
                default_hi = min(y_max, float(prev_hi))
                if default_lo > default_hi:
                    default_lo, default_hi = y_min, y_max

                sel_years_range = st.slider(
                    "YEARS_OF_WORK",
                    min_value=int(y_min) if step_is_int else y_min,
                    max_value=int(y_max) if step_is_int else y_max,
                    value=(int(default_lo), int(default_hi))
                    if step_is_int
                    else (default_lo, default_hi),
                    step=step,
                    key="sel_years_range",
                )

                hi = float(sel_years_range[1])
                include_125 = (
                    int(hi) == int(y_max)
                    if step_is_int
                    else np.isclose(hi, y_max, rtol=0, atol=1e-9)
                )
            else:
                sel_years_range = None
                include_125 = True

        sel_years_from_date = None
        if "DATE" in _base.columns:
            _dates = pd.to_datetime(_base["DATE"], errors="coerce")
            year_opts = sorted(_dates.dropna().dt.year.unique().tolist())
            if year_opts:
                prev_years = st.session_state.get("sel_years_from_date", year_opts)
                sel_years_from_date = st.multiselect(
                    "YEAR",
                    options=year_opts,
                    default=[y for y in prev_years if y in year_opts] or year_opts,
                    key="sel_years_from_date",
                )

        selections = dict(
            sel_depo_groups=sel_depo_groups,
            sel_depos=sel_depos,
            sel_gender=sel_gender,
            sel_job=sel_job,
            sel_years_range=sel_years_range,
            include_125=include_125,
            sel_years_from_date=sel_years_from_date,
        )
        return selections


def apply_filters(df_in: pd.DataFrame, F: dict) -> pd.DataFrame:
    out = df_in.copy()
    if F["sel_depo_groups"] is not None and "DEPO_GROUP" in out.columns:
        out = out[out["DEPO_GROUP"].isin(F["sel_depo_groups"])]
    if F["sel_depos"] is not None and "DEPO" in out.columns:
        out = out[out["DEPO"].isin(F["sel_depos"])]
    if F["sel_gender"] is not None and "GENDER" in out.columns:
        out = out[out["GENDER"].isin(F["sel_gender"])]
    if F["sel_job"] is not None and "JOB_TYPE" in out.columns:
        out = out[out["JOB_TYPE"].isin(F["sel_job"])]
    if F["sel_years_range"] is not None and "YEARS_OF_WORK" in out.columns:
        yrs = pd.to_numeric(out["YEARS_OF_WORK"], errors="coerce")
        lo, hi = float(F["sel_years_range"][0]), float(F["sel_years_range"][1])
        keep = yrs.between(lo, hi, inclusive="both")
        if F["include_125"] and "YEARS_OF_WORK" in out.columns:
            keep |= (yrs == SENTINEL_YEARS)
        out = out[keep]
    return out


# =============================================================================
# HISTOGRAMS (with AN_PH / AN_ST split toggles)
# =============================================================================
# (keep your existing render_histograms function unchanged)
# -----------------------------------------------------------------------------
def render_histograms(df_f: pd.DataFrame):
    st.subheader("Distributions")

    # Copy + normalise gender
    df_hist = df_f.copy()
    if "GENDER" in df_hist.columns:
        g = df_hist["GENDER"].astype("string").str.strip().str.lower()
        df_hist["GENDER_N"] = (
            g.map({"m": "Male", "male": "Male", "f": "Female", "female": "Female"})
            .astype("string")
        )
    else:
        df_hist["GENDER_N"] = pd.Series(pd.NA, index=df_hist.index, dtype="string")

    # Toggles
    hist_pct = st.toggle("Show histogram y-axis as %", value=False, key="hist_pct")
    gender_split = st.toggle("Split histograms by gender (M/F)", value=False, key="hist_gender_split")
    facet_by_grade = st.toggle("Small multiples by JOB_TYPE", value=False, key="facet_by_grade")
    anph_split = st.toggle("Split histograms by PH (True/False)", value=False, key="hist_anph_split")
    st_split = st.toggle("Split histograms by ST (True/False)", value=False, key="hist_st_split")  # NEW

    # Normalise AN_PH to labelled groups (no filtering)
    if "AN_PH" in df_hist.columns:
        col = df_hist["AN_PH"]
        if pd.api.types.is_bool_dtype(col):
            anph_bool = col
        else:
            anph_bool = (
                col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})
            )
        df_hist["AN_PH_N"] = anph_bool.map({True: "AN_PH=True", False: "AN_PH=False"})
    else:
        df_hist["AN_PH_N"] = pd.Series(pd.NA, index=df_hist.index, dtype="string")

    # NEW: Normalise AN_ST to labelled groups (no filtering)
    if "AN_ST" in df_hist.columns:
        col = df_hist["AN_ST"]
        if pd.api.types.is_bool_dtype(col):
            anst_bool = col
        else:
            anst_bool = (
                col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})
            )
        df_hist["AN_ST_N"] = anst_bool.map({True: "AN_ST=True", False: "AN_ST=False"})
    else:
        df_hist["AN_ST_N"] = pd.Series(pd.NA, index=df_hist.index, dtype="string")

    # Split precedence: ST > PH > Gender > None
    split_mode = (
        "st" if st_split and df_hist["AN_ST_N"].notna().any()
        else "anph" if anph_split and df_hist["AN_PH_N"].notna().any()
        else ("gender" if gender_split else None)
    )

    COLORS = {"Male": "#1f77b4", "Female": "#e377c2"}
    ANPH_COLORS = {"AN_PH=True": "#2ca02c", "AN_PH=False": "#ff7f0e"}
    ANST_COLORS = {"AN_ST=True": "#9467bd", "AN_ST=False": "#8c564b"}  # NEW

    # -------------------------------------------------------------------------
    # FACET VIEW (by JOB_TYPE)
    # -------------------------------------------------------------------------
    if facet_by_grade and "JOB_TYPE" in df_hist.columns and not df_hist.empty:
        # A) Block size distribution
        cats = [str(i) for i in range(1, 21)] + [">20"]
        recs = []
        for grade in sorted(df_hist["JOB_TYPE"].dropna().unique().tolist()):
            d_g = df_hist[df_hist["JOB_TYPE"] == grade]
            s = pd.to_numeric(d_g["BLOCK_SIZE"], errors="coerce")
            s = s[s.notna() & (s > 0)]
            if s.empty:
                continue

            bucket = np.where(s > 20, ">20", s.clip(upper=20).astype(int).astype(str))
            if split_mode == "anph":
                grp = d_g.loc[s.index, "AN_PH_N"].fillna("Unknown")
            elif split_mode == "st":  # NEW
                grp = d_g.loc[s.index, "AN_ST_N"].fillna("Unknown")
            elif split_mode == "gender":
                grp = d_g.loc[s.index, "GENDER_N"].fillna("Unknown")
            else:
                grp = pd.Series(["All"] * len(s), index=s.index, dtype="object")

            df_tmp = pd.DataFrame({"bucket": bucket, "group": grp, "JOB_TYPE": str(grade)})
            counts = (
                df_tmp.groupby(["JOB_TYPE", "group", "bucket"])
                .size()
                .rename("value")
                .reset_index()
            )
            counts["bucket"] = pd.Categorical(counts["bucket"], categories=cats, ordered=True)
            counts = counts.sort_values(["JOB_TYPE", "group", "bucket"])

            if hist_pct:
                counts["value"] = counts.groupby(["JOB_TYPE", "group"])["value"].transform(
                    lambda v: (v / v.sum() * 100.0) if v.sum() else v
                )
            recs.append(counts)

        if recs:
            df_bsize = pd.concat(recs, ignore_index=True)
            fig_grid1 = px.bar(
                df_bsize,
                x="bucket",
                y="value",
                facet_col="JOB_TYPE",
                facet_col_wrap=4,
                color="group" if split_mode else None,
                category_orders={"bucket": cats},
                labels={
                    "bucket": "Block size (1–20, >20)",
                    "value": "% of blocks" if hist_pct else "Count",
                },
                title="Distribution of Block Size — by JOB_TYPE",
            )
            fig_grid1.for_each_yaxis(lambda a: a.update(matches="y"))
            if split_mode in {"anph", "st", "gender"}:
                fig_grid1.update_traces(opacity=0.6)
                fig_grid1.update_layout(barmode="overlay")
                if split_mode == "anph":
                    fig_grid1.for_each_trace(
                        lambda t: t.update(marker_color=ANPH_COLORS.get(t.name, None))
                    )
                elif split_mode == "st":  # NEW
                    fig_grid1.for_each_trace(
                        lambda t: t.update(marker_color=ANST_COLORS.get(t.name, None))
                    )
            fig_grid1.update_layout(bargap=0.05, margin=dict(l=60, r=40, t=60, b=50))
            if hist_pct and not df_bsize.empty:
                ymax = min(100.0, max(1.0, float(df_bsize["value"].max()) * 1.08))
                fig_grid1.layout.yaxis.update(range=[0, ymax])
            st.plotly_chart(fig_grid1, use_container_width=True)

        # B) AN blocks per person
        grade_per_name = (
            df_hist.groupby("NAME")["JOB_TYPE"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iloc[0] if s.dropna().size else pd.NA))
        )
        blocks_per_person = (
            df_hist.groupby("NAME", as_index=False).size().rename(columns={"size": "NUM_BLOCKS"})
        ).merge(grade_per_name.reset_index(), on="NAME", how="left")

        recs2 = []
        if not blocks_per_person.empty:
            if split_mode == "anph":
                group_per_name = (
                    df_hist.groupby("NAME")["AN_PH_N"]
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iloc[0] if s.dropna().size else "Unknown"))
                )
            elif split_mode == "st":
                group_per_name = (
                    df_hist.groupby("NAME")["AN_ST_N"]
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iloc[0] if s.dropna().size else "Unknown"))
                )
            elif split_mode == "gender":
                group_per_name = (
                    df_hist.groupby("NAME")["GENDER_N"]
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iloc[0] if s.dropna().size else "Unknown"))
                )
            else:
                group_per_name = pd.Series("All", index=blocks_per_person["NAME"].unique(), dtype="object")

            bpp2 = blocks_per_person.merge(
                group_per_name.rename("GROUP").reset_index(), on="NAME", how="left"
            )
            max_nb = int(bpp2["NUM_BLOCKS"].max()) if not bpp2.empty else 0

            for grade in sorted(bpp2["JOB_TYPE"].dropna().unique().tolist()):
                d = bpp2[bpp2["JOB_TYPE"] == grade]
                if d.empty:
                    continue
                if split_mode:
                    for grp_name, d_grp in d.groupby("GROUP"):
                        vc = d_grp["NUM_BLOCKS"].value_counts().sort_index()
                        if max_nb > 0:
                            vc = vc.reindex(range(1, max_nb + 1), fill_value=0)
                        y = (vc / vc.sum() * 100.0) if (hist_pct and vc.sum()) else vc
                        for k, v in y.items():
                            recs2.append(
                                {
                                    "JOB_TYPE": str(grade),
                                    "group": str(grp_name),
                                    "num_blocks": int(k),
                                    "value": float(v),
                                }
                            )
                else:
                    vc = d["NUM_BLOCKS"].value_counts().sort_index()
                    if max_nb > 0:
                        vc = vc.reindex(range(1, max_nb + 1), fill_value=0)
                    y = (vc / vc.sum() * 100.0) if (hist_pct and vc.sum()) else vc
                    for k, v in y.items():
                        recs2.append(
                            {"JOB_TYPE": str(grade), "group": "All", "num_blocks": int(k), "value": float(v)}
                        )

        if recs2:
            df_numblocks = pd.DataFrame(recs2)
            fig_grid2 = px.bar(
                df_numblocks,
                x="num_blocks",
                y="value",
                facet_col="JOB_TYPE",
                facet_col_wrap=4,
                color="group" if split_mode else None,
                labels={
                    "num_blocks": "AN blocks per person (year)",
                    "value": "% of employees" if hist_pct else "Employees",
                },
                title="Distribution of AN Blocks per Person — by JOB_TYPE",
            )
            fig_grid2.for_each_yaxis(lambda a: a.update(matches="y"))
            if split_mode in {"anph", "st", "gender"}:
                fig_grid2.update_traces(opacity=0.6)
                fig_grid2.update_layout(barmode="overlay")
                if split_mode == "anph":
                    fig_grid2.for_each_trace(lambda t: t.update(marker_color=ANPH_COLORS.get(t.name, None)))
                elif split_mode == "st":  # NEW
                    fig_grid2.for_each_trace(lambda t: t.update(marker_color=ANST_COLORS.get(t.name, None)))
            fig_grid2.update_layout(bargap=0.05, margin=dict(l=60, r=40, t=60, b=50))
            if not df_numblocks.empty:
                fig_grid2.update_xaxes(tickmode="linear", dtick=1)
            if hist_pct and not df_numblocks.empty:
                ymax2 = min(100.0, max(1.0, float(df_numblocks["value"].max()) * 1.08))
                fig_grid2.layout.yaxis.update(range=[0, ymax2])
            st.plotly_chart(fig_grid2, use_container_width=True)

        return  # end facet branch

    # -------------------------------------------------------------------------
    # ORIGINAL TWO HISTOGRAMS (single view)
    # -------------------------------------------------------------------------
    if {"NAME", "BLOCK_SIZE"}.issubset(df_hist.columns) and not df_hist.empty:
        blocks_per_person = (
            df_hist.groupby("NAME", as_index=False)
            .size()
            .rename(columns={"size": "NUM_BLOCKS"})
        )

        c1, c2 = st.columns(2)

        # Left: Distribution of Block Size (bucketed 1..20, >20)
        with c1:
            cats = [str(i) for i in range(1, 21)] + [">20"]

            if not split_mode:
                s = pd.to_numeric(df_hist["BLOCK_SIZE"], errors="coerce")
                s = s[s.notna() & (s > 0)]
                bucket = np.where(s > 20, ">20", s.clip(upper=20).astype(int).astype(str))
                counts = pd.Series(bucket, dtype="object").value_counts().reindex(cats, fill_value=0)
                if hist_pct:
                    y = (counts / counts.sum() * 100) if counts.sum() else counts.astype(float)
                    y_title = "% of blocks"
                else:
                    y = counts.values
                    y_title = "Count"
                fig_h1 = go.Figure(go.Bar(x=cats, y=y, name="All blocks"))

            elif split_mode == "gender":
                fig_h1 = go.Figure()
                for gname in ["Male", "Female"]:
                    d_g = df_hist[df_hist["GENDER_N"] == gname]
                    if d_g.empty:
                        continue
                    s = pd.to_numeric(d_g["BLOCK_SIZE"], errors="coerce")
                    s = s[s.notna() & (s > 0)]
                    bucket = np.where(s > 20, ">20", s.clip(upper=20).astype(int).astype(str))
                    counts = pd.Series(bucket, dtype="object").value_counts().reindex(cats, fill_value=0)
                    y = (counts / counts.sum() * 100) if (hist_pct and counts.sum()) else counts
                    fig_h1.add_bar(
                        x=cats, y=y.values.tolist(), name=gname, opacity=0.6, marker_color=COLORS.get(gname, None)
                    )
                fig_h1.update_layout(barmode="overlay")
                y_title = "% of blocks" if hist_pct else "Count"

            elif split_mode == "anph":
                fig_h1 = go.Figure()
                for gname in ["AN_PH=True", "AN_PH=False"]:
                    d_g = df_hist[df_hist["AN_PH_N"] == gname]
                    if d_g.empty:
                        continue
                    s = pd.to_numeric(d_g["BLOCK_SIZE"], errors="coerce")
                    s = s[s.notna() & (s > 0)]
                    bucket = np.where(s > 20, ">20", s.clip(upper=20).astype(int).astype(str))
                    counts = pd.Series(bucket, dtype="object").value_counts().reindex(cats, fill_value=0)
                    y = (counts / counts.sum() * 100) if (hist_pct and counts.sum()) else counts
                    fig_h1.add_bar(
                        x=cats, y=y.values.tolist(), name=gname, opacity=0.6, marker_color=ANPH_COLORS.get(gname, None)
                    )
                fig_h1.update_layout(barmode="overlay")
                y_title = "% of blocks" if hist_pct else "Count"

            else:  # split_mode == "st"
                fig_h1 = go.Figure()
                for gname in ["AN_ST=True", "AN_ST=False"]:
                    d_g = df_hist[df_hist["AN_ST_N"] == gname]
                    if d_g.empty:
                        continue
                    s = pd.to_numeric(d_g["BLOCK_SIZE"], errors="coerce")
                    s = s[s.notna() & (s > 0)]
                    bucket = np.where(s > 20, ">20", s.clip(upper=20).astype(int).astype(str))
                    counts = pd.Series(bucket, dtype="object").value_counts().reindex(cats, fill_value=0)
                    y = (counts / counts.sum() * 100) if (hist_pct and counts.sum()) else counts
                    fig_h1.add_bar(
                        x=cats, y=y.values.tolist(), name=gname, opacity=0.6,
                        marker_color=ANST_COLORS.get(gname, None)
                    )
                fig_h1.update_layout(barmode="overlay")
                y_title = "% of blocks" if hist_pct else "Count"

            fig_h1.update_layout(
                title="Distribution of Block Size (days)",
                xaxis_title="Block size (1–20, >20)",
                yaxis_title=y_title,
                bargap=0.05,
                margin=dict(l=60, r=40, t=60, b=50),
                xaxis=dict(type="category"),
            )
            st.plotly_chart(fig_h1, use_container_width=True)

        # Right: Distribution of AN Blocks per Person
        with c2:
            if "GENDER_N" in df_hist.columns:
                gender_per_name = (
                    df_hist.groupby("NAME")["GENDER_N"]
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iloc[0] if s.dropna().size else pd.NA))
                )
                bpp = (
                    df_hist.groupby("NAME", as_index=False)
                    .size()
                    .rename(columns={"size": "NUM_BLOCKS"})
                    .merge(gender_per_name.reset_index(), on="NAME", how="left")
                )
            else:
                bpp = blocks_per_person.assign(
                    GENDER_N=pd.Series(pd.NA, index=blocks_per_person.index, dtype="string")
                )

            max_nb = int(bpp["NUM_BLOCKS"].max()) if not bpp.empty else 0

            if not split_mode:
                fig_h2 = go.Figure(
                    go.Histogram(
                        x=blocks_per_person["NUM_BLOCKS"],
                        xbins=dict(start=1, end=max_nb + 1, size=1) if max_nb > 0 else None,
                        histnorm="percent" if hist_pct else None,
                        name="All employees",
                    )
                )

            elif split_mode == "gender":
                fig_h2 = go.Figure()
                for gname in ["Male", "Female"]:
                    d = bpp[bpp["GENDER_N"] == gname]
                    if d.empty:
                        continue
                    fig_h2.add_histogram(
                        x=d["NUM_BLOCKS"],
                        xbins=dict(start=1, end=max_nb + 1, size=1) if max_nb > 0 else None,
                        histnorm="percent" if hist_pct else None,
                        opacity=0.6,
                        name=gname,
                        marker_color=COLORS.get(gname, None),
                    )
                fig_h2.update_layout(barmode="overlay")

            elif split_mode == "anph":
                fig_h2 = go.Figure()
                anph_per_name = (
                    df_hist.groupby("NAME")["AN_PH_N"]
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iloc[0] if s.dropna().size else pd.NA))
                )
                bpp_anph = blocks_per_person.merge(
                    anph_per_name.rename("AN_PH_N").reset_index(), on="NAME", how="left"
                )
                for gname in ["AN_PH=True", "AN_PH=False"]:
                    d = bpp_anph[bpp_anph["AN_PH_N"] == gname]
                    if d.empty:
                        continue
                    fig_h2.add_histogram(
                        x=d["NUM_BLOCKS"],
                        xbins=dict(start=1, end=max_nb + 1, size=1) if max_nb > 0 else None,
                        histnorm="percent" if hist_pct else None,
                        opacity=0.6,
                        name=gname,
                        marker_color=ANPH_COLORS.get(gname, None),
                    )
                fig_h2.update_layout(barmode="overlay")

            else:  # split_mode == "st"
                fig_h2 = go.Figure()
                anst_per_name = (
                    df_hist.groupby("NAME")["AN_ST_N"]
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iloc[0] if s.dropna().size else pd.NA))
                )
                bpp_st = blocks_per_person.merge(
                    anst_per_name.rename("AN_ST_N").reset_index(), on="NAME", how="left"
                )
                for gname in ["AN_ST=True", "AN_ST=False"]:
                    d = bpp_st[bpp_st["AN_ST_N"] == gname]
                    if d.empty:
                        continue
                    fig_h2.add_histogram(
                        x=d["NUM_BLOCKS"],
                        xbins=dict(start=1, end=max_nb + 1, size=1) if max_nb > 0 else None,
                        histnorm="percent" if hist_pct else None,
                        opacity=0.6,
                        name=gname,
                        marker_color=ANST_COLORS.get(gname, None),
                    )
                fig_h2.update_layout(barmode="overlay")

            fig_h2.update_layout(
                title="Distribution of AN Blocks per Person",
                xaxis_title="AN blocks per person (year)",
                yaxis_title="% of employees" if hist_pct else "Employees",
                bargap=0.05,
                margin=dict(l=60, r=40, t=60, b=50),
            )

            # Centre integer ticks visually
            if max_nb > 0:
                tickvals = [i + 0.5 for i in range(1, max_nb + 1)]
                ticktext = list(range(1, max_nb + 1))
                fig_h2.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

            st.plotly_chart(fig_h2, use_container_width=True)

    else:
        st.info("CSV must include 'NAME' and 'BLOCK_SIZE' to draw the histograms.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    st.title("Employee AN Dashboard")

    # Load + clean (main CSV)
    try:
        df = load_csv_upper(DEFAULT_PL_CSV)
        df = coerce_blocksize(df)
        df = coerce_if_any_pl01(df)
        df = coerce_if_any_st01(df)  # no-op if IF_ANY_ST is absent
    except Exception as e:
        st.error(f"Couldn't read or parse the main CSV.\n\n{e}")
        st.stop()

    # Sidebar filters → selections → apply
    filters = sidebar_filter_controls(df)
    df_f = apply_filters(df, filters)

    # Histograms
    # (unchanged)
    # If you had render_histograms(df_f), keep it:
    render_histograms(df_f)

    # Heatmaps
    tab_all, tab_pl = st.tabs(["AN blocks (all)", "PL blocks (IF_ANY_PL == 1)"])
    with tab_all:
        if {"NAME", "BLOCK_SIZE"}.issubset(df_f.columns):
            pivot_all, _ = pivot_allblocks(df_f)
            fig_all = make_heatmap(
                pivot_all,
                title="Employees by (Number of AN Blocks × Block Size) — All blocks",
                y_title="AN blocks per person (per year)",
            )
            st.plotly_chart(fig_all, use_container_width=True)

            # ---------- NEW (DIL heatmap right under the first one) ----------
            try:
                df_dil = load_dil_csv(DEFAULT_DIL_CSV)
                # We only need BLOCK_SIZE and DIL_BLOCK_SIZE; no filters shared from df_f.
                pivot_dil = pivot_dil_blocks(df_dil)
                if not pivot_dil.empty:
                    fig_dil = make_heatmap(
                        pivot_dil,
                        title="Employees by (DIL Block Size × AN Block Size)",
                        y_title="DIL block size (days)",
                    )
                    st.plotly_chart(fig_dil, use_container_width=True)
                else:
                    st.info("No rows in DIL file after coercion; cannot draw DIL heatmap.")
            except Exception as e:
                st.warning(f"Couldn't build the DIL heatmap: {e}")
            # -----------------------------------------------------------------

            # Existing PH-only and ST-only heatmaps (unchanged)
            if "AN_PH" in df_f.columns:
                col = df_f["AN_PH"]
                mask_true = col if pd.api.types.is_bool_dtype(col) else col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})
                df_anph_true = df_f[mask_true == True].copy()
                if not df_anph_true.empty:
                    pivot_true, _ = pivot_allblocks(df_anph_true)
                    fig_true = make_heatmap(
                        pivot_true,
                        title="Employees by (Number of AN Blocks × Block Size) — Public Holidays Only",
                        y_title="AN blocks per person (per year)",
                    )
                    st.plotly_chart(fig_true, use_container_width=True)
                else:
                    st.info("No rows with AN_PH == TRUE after filters.")
            else:
                st.info("CSV has no 'AN_PH' column — cannot build AN_PH==TRUE heatmap.")

            if "AN_ST" in df_f.columns:
                col = df_f["AN_ST"]
                mask_st_true = col if pd.api.types.is_bool_dtype(col) else col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})
                df_anst_true = df_f[mask_st_true == True].copy()
                if not df_anst_true.empty:
                    pivot_st, _ = pivot_allblocks(df_anst_true)
                    fig_st = make_heatmap(
                        pivot_st,
                        title="Employees by (Number of AN Blocks × Block Size) — Shutdowns Only",
                        y_title="AN blocks per person (per year)",
                    )
                    st.plotly_chart(fig_st, use_container_width=True)
                else:
                    st.info("No rows with AN_ST == TRUE after filters.")
            else:
                st.info("CSV has no 'AN_ST' column — cannot build AN_ST==TRUE heatmap.")
        else:
            st.error("CSV must contain 'NAME' and 'BLOCK_SIZE' for the AN heatmap.")

    with tab_pl:
        if "IF_ANY_PL" not in df_f.columns:
            st.info("CSV has no 'IF_ANY_PL' column — cannot build PL-only heatmap.")
        else:
            df_pl_only = df_f[df_f["IF_ANY_PL"] == 1].copy()
            if df_pl_only.empty:
                st.info("No rows with IF_ANY_PL == 1 after filters.")
            elif "NAME" not in df_pl_only.columns:
                st.error("CSV must contain 'NAME' for the PL-only heatmap.")
            else:
                pivot_pl, _ = pivot_plonly(df_pl_only)
                fig_pl = make_heatmap(
                    pivot_pl,
                    title="Employees with PL by (PL Blocks per Person × Block Size) — PL only",
                    y_title="PL blocks per person (per year)",
                )
                st.plotly_chart(fig_pl, use_container_width=True)

    # Snapshot
    with st.expander("Data snapshot (first 50 rows)"):
        st.dataframe(df_f.head(50))

        # PL rate charts (overall / PH-only / ST-only)
    st.markdown("---")
    st.subheader("PL rate vs Block Size")
    bars_as_percent = st.toggle("Show bars as % of total", value=True)
    if {"BLOCK_SIZE", "IF_ANY_PL"}.issubset(df_f.columns):
        if df_f.empty:
            st.info("No data after filters to draw the PL rate chart.")
        else:
            fig_rate = make_pl_rate_chart(df_f, bars_as_percent=bars_as_percent)
            if fig_rate.data:
                st.plotly_chart(fig_rate, use_container_width=True)
            else:
                st.info("Not enough data to render the PL rate chart.")
    else:
        st.info("CSV must include 'BLOCK_SIZE' and 'IF_ANY_PL' to render the PL rate chart.")

    st.subheader("PL rate vs Block Size — Public holidays only")
    # st.dataframe(df_f)
    bars_as_percent_ph = st.toggle("Show bars as % of total (PH only)", value=True, key="bars_as_percent_ph")
    fig_rate_ph = make_pl_rate_chart_anph_true(df_f, bars_as_percent=bars_as_percent_ph)
    if fig_rate_ph.data:
        st.plotly_chart(fig_rate_ph, use_container_width=True)
    else:
        st.info("No PH-only data available after filters.")

    st.subheader("PL rate vs Block Size — Shutdowns only")
    bars_as_percent_st = st.toggle("Show bars as % of total (ST only)", value=True, key="bars_as_percent_st")
    fig_rate_st = make_pl_rate_chart_anst_true(df_f, bars_as_percent=bars_as_percent_st)
    if fig_rate_st.data:
        st.plotly_chart(fig_rate_st, use_container_width=True)
    else:
        st.info("No ST-only data available after filters.")


if __name__ == "__main__":
    main()
