# streamlit_app_ph_st.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Leave Granularity — PH/ST", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_FOLDER = PROJECT_ROOT / "csv"

DEFAULT_PL_CSV = CSV_FOLDER / "an_pl_ph_data.csv"
SENTINEL_YEARS = 125

# =============================================================================
# HELPERS (DATA LOADING / CLEANING)
# =============================================================================
@st.cache_data
def load_csv_upper(path: str) -> pd.DataFrame:
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
# HEATMAP HELPERS
# =============================================================================
def make_heatmap(pivot: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    numeric = pivot.apply(pd.to_numeric, errors="coerce").fillna(0)
    row_has_values = numeric.gt(0).any(axis=1)
    data = pivot.loc[row_has_values] if row_has_values.any() else pivot
    x_labels = [str(x) for x in data.columns]
    y_labels = [str(y) for y in data.index]
    fig = go.Figure(
        data=go.Heatmap(
            z=data.values, x=x_labels, y=y_labels,
            colorscale="Viridis", colorbar=dict(title="Employees"),
            hovertemplate="Block size: %{x}<br>%{y}<br>Employees: %{z}<extra></extra>",
            zmin=0,
        )
    )
    fig.update_xaxes(
        title="Block size (days)", type="category",
        categoryorder="array", categoryarray=x_labels,
        tickmode="array", tickvals=x_labels, ticktext=x_labels, dtick=1,
    )
    fig.update_yaxes(
        autorange="reversed", title=y_title, type="category",
        categoryorder="array", categoryarray=y_labels, dtick=1,
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
    df_uniq["BLOCK_SIZE_BUCKET"] = df_uniq["BLOCK_SIZE"].where(df_uniq["BLOCK_SIZE"] <= 30, ">30")
    df_uniq["NUM_BLOCKS_BUCKET"] = df_uniq["NUM_BLOCKS"].where(df_uniq["NUM_BLOCKS"] <= 10, ">10")

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

# =============================================================================
# SIDEBAR FILTERS (same as before; PH/ST behaviour driven in main charts)
# =============================================================================
def sidebar_filter_controls(df: pd.DataFrame):
    with st.sidebar:
        st.header("Filters")

        sel_depo_groups = None
        if "DEPO_GROUP" in df.columns:
            opts = sorted(df["DEPO_GROUP"].dropna().unique().tolist())
            sel_depo_groups = st.multiselect(
                "DEPO_GROUP", options=opts,
                default=st.session_state.get("sel_depo_groups", opts) if opts else None,
                key="sel_depo_groups",
            )

        base = df.copy()
        if sel_depo_groups is not None and "DEPO_GROUP" in base.columns:
            base = base[base["DEPO_GROUP"].isin(sel_depo_groups)]

        sel_depos = None
        if "DEPO" in base.columns:
            opts = sorted(base["DEPO"].dropna().unique().tolist())
            prev = st.session_state.get("sel_depos", opts)
            sel_depos = st.multiselect(
                "DEPO",
                options=opts,
                default=[d for d in prev if d in opts] or opts if opts else None,
                key="sel_depos",
            )
            base = base[base["DEPO"].isin(sel_depos)] if sel_depos is not None else base

        sel_gender = None
        if "GENDER" in base.columns:
            opts = sorted(base["GENDER"].dropna().unique().tolist())
            prev = st.session_state.get("sel_gender", opts)
            sel_gender = st.multiselect(
                "GENDER", options=opts,
                default=[g for g in prev if g in opts] or opts if opts else None,
                key="sel_gender",
            )

        sel_job = None
        if "JOB_TYPE" in base.columns:
            opts = sorted(base["JOB_TYPE"].dropna().unique().tolist())
            prev = st.session_state.get("sel_job", opts)
            sel_job = st.multiselect(
                "JOB_TYPE", options=opts,
                default=[g for g in prev if g in opts] or opts if opts else None,
                key="sel_job",
            )

        sel_years_range = None
        include_125 = False
        if "YEARS_OF_WORK" in base.columns:
            years_numeric = pd.to_numeric(base["YEARS_OF_WORK"], errors="coerce").dropna()
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
                    value=(int(default_lo), int(default_hi)) if step_is_int else (default_lo, default_hi),
                    step=step,
                    key="sel_years_range",
                )
                hi = float(sel_years_range[1])
                include_125 = int(hi) == int(y_max) if step_is_int else np.isclose(hi, y_max)
            else:
                sel_years_range, include_125 = None, True

        sel_years_from_date = None
        if "DATE" in base.columns:
            _dates = pd.to_datetime(base["DATE"], errors="coerce")
            year_opts = sorted(_dates.dropna().dt.year.unique().tolist())
            if year_opts:
                prev_years = st.session_state.get("sel_years_from_date", year_opts)
                sel_years_from_date = st.multiselect(
                    "YEAR", options=year_opts,
                    default=[y for y in prev_years if y in year_opts] or year_opts,
                    key="sel_years_from_date",
                )

        return dict(
            sel_depo_groups=sel_depo_groups,
            sel_depos=sel_depos,
            sel_gender=sel_gender,
            sel_job=sel_job,
            sel_years_range=sel_years_range,
            include_125=include_125,
            sel_years_from_date=sel_years_from_date,
        )

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
        if F["include_125"]:
            keep |= (yrs == SENTINEL_YEARS)
        out = out[keep]
    if F["sel_years_from_date"] is not None and "DATE" in out.columns:
        _dates = pd.to_datetime(out["DATE"], errors="coerce")
        out = out[_dates.dt.year.isin(F["sel_years_from_date"])]
    return out

# =============================================================================
# HISTOGRAMS (keep first toggle + PH/ST toggles only)
# =============================================================================
def render_histograms(df_f: pd.DataFrame):
    st.subheader("Distributions")

    df_hist = df_f.copy()
    # FIRST TOGGLE
    hist_pct = st.toggle("Show histogram y-axis as %", value=False, key="hist_pct")
    # LAST TWO TOGGLES (PH/ST)
    anph_split = st.toggle("Split histograms by PH (True/False)", value=False, key="hist_anph_split")
    st_split = st.toggle("Split histograms by ST (True/False)", value=False, key="hist_st_split")

    # Normalise AN_PH
    if "AN_PH" in df_hist.columns:
        col = df_hist["AN_PH"]
        anph_bool = col if pd.api.types.is_bool_dtype(col) else col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})
        df_hist["AN_PH_N"] = anph_bool.map({True: "AN_PH=True", False: "AN_PH=False"})
    else:
        df_hist["AN_PH_N"] = pd.Series(pd.NA, index=df_hist.index, dtype="string")

    # Normalise AN_ST
    if "AN_ST" in df_hist.columns:
        col = df_hist["AN_ST"]
        anst_bool = col if pd.api.types.is_bool_dtype(col) else col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})
        df_hist["AN_ST_N"] = anst_bool.map({True: "AN_ST=True", False: "AN_ST=False"})
    else:
        df_hist["AN_ST_N"] = pd.Series(pd.NA, index=df_hist.index, dtype="string")

    # Split precedence: ST > PH > None
    split_mode = (
        "st" if st_split and df_hist["AN_ST_N"].notna().any()
        else "anph" if anph_split and df_hist["AN_PH_N"].notna().any()
        else None
    )

    ANPH_COLORS = {"AN_PH=True": "#2ca02c", "AN_PH=False": "#ff7f0e"}
    ANST_COLORS = {"AN_ST=True": "#9467bd", "AN_ST=False": "#8c564b"}

    if {"NAME", "BLOCK_SIZE"}.issubset(df_hist.columns) and not df_hist.empty:
        blocks_per_person = (
            df_hist.groupby("NAME", as_index=False)
            .size()
            .rename(columns={"size": "NUM_BLOCKS"})
        )
        c1, c2 = st.columns(2)

        # Left: block size distribution
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
                    y, y_title = counts.values, "Count"
                fig_h1 = go.Figure(go.Bar(x=cats, y=y, name="All blocks"))
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
                    fig_h1.add_bar(x=cats, y=y.values.tolist(), name=gname, opacity=0.6,
                                   marker_color=ANPH_COLORS.get(gname, None))
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
                    fig_h1.add_bar(x=cats, y=y.values.tolist(), name=gname, opacity=0.6,
                                   marker_color=ANST_COLORS.get(gname, None))
                fig_h1.update_layout(barmode="overlay")
                y_title = "% of blocks" if hist_pct else "Count"

            fig_h1.update_layout(
                title="Distribution of Block Size (days)",
                xaxis_title="Block size (1–20, >20)", yaxis_title=y_title,
                bargap=0.05, margin=dict(l=60, r=40, t=60, b=50), xaxis=dict(type="category"),
            )
            st.plotly_chart(fig_h1, use_container_width=True)

        # Right: blocks per person
        with c2:
            max_nb = int(blocks_per_person["NUM_BLOCKS"].max()) if not blocks_per_person.empty else 0
            if not split_mode:
                fig_h2 = go.Figure(
                    go.Histogram(
                        x=blocks_per_person["NUM_BLOCKS"],
                        xbins=dict(start=1, end=max_nb + 1, size=1) if max_nb > 0 else None,
                        histnorm="percent" if hist_pct else None,
                        name="All employees",
                    )
                )
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
                        opacity=0.6, name=gname,
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
                        opacity=0.6, name=gname,
                        marker_color=ANST_COLORS.get(gname, None),
                    )
                fig_h2.update_layout(barmode="overlay")

            fig_h2.update_layout(
                title="Distribution of AN Blocks per Person",
                xaxis_title="AN blocks per person (year)",
                yaxis_title="% of employees" if hist_pct else "Employees",
                bargap=0.05, margin=dict(l=60, r=40, t=60, b=50),
            )
            if max_nb > 0:
                tickvals = [i + 0.5 for i in range(1, max_nb + 1)]
                ticktext = list(range(1, max_nb + 1))
                fig_h2.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            st.plotly_chart(fig_h2, use_container_width=True)
    else:
        st.info("CSV must include 'NAME' and 'BLOCK_SIZE' to draw the histograms.")

# =============================================================================
# PH/ST-ONLY PL RATE CHARTS
# =============================================================================
def _pl_rate_core(d: pd.DataFrame, bars_as_percent: bool, left_title_prefix: str, line_name: str, overall_title: str) -> go.Figure:
    size = pd.to_numeric(d["BLOCK_SIZE"], errors="coerce")
    d = d.loc[size.notna() & (size > 0)].copy()
    if d.empty:
        return go.Figure()
    size = size.loc[d.index]
    d["SIZE_BUCKET"] = np.where(size > 15, ">15", size.clip(upper=15).astype(int).astype(str))

    if set(pd.unique(d["IF_ANY_PL"].dropna())) - {0, 1}:
        s = pd.to_numeric(d["IF_ANY_PL"], errors="coerce").fillna(0)
        d["IF_ANY_PL"] = (s > 0).astype(int)

    ct = d.groupby(["SIZE_BUCKET", "IF_ANY_PL"]).size().unstack()
    for col_id in (0, 1):
        if col_id not in ct.columns:
            ct[col_id] = 0
    ct = ct[[0, 1]].rename(columns={0: "no_pl", 1: "pl"}).fillna(0)

    ct["n"] = ct["no_pl"] + ct["pl"]
    total_n = ct["n"].sum()
    if total_n == 0:
        return go.Figure()

    ct["pct_total"] = (ct["n"] / total_n * 100)
    ct["pl_rate"] = (ct["pl"] / ct["n"] * 100)

    cats = [str(i) for i in range(1, 16)] + [">15"]
    ct = ct.reindex(cats)

    if bars_as_percent:
        y_bars = ct["pct_total"].fillna(0)
        bar_name = f"% of {left_title_prefix} blocks"
        bar_hover = f"Block size %{{x}}<br>% of {left_title_prefix} total %{{y:.1f}}%<extra></extra>"
        left_title = f"% of {left_title_prefix} blocks"
    else:
        y_bars = ct["n"].fillna(0)
        bar_name = f"# {left_title_prefix} blocks"
        bar_hover = f"Block size %{{x}}<br>Count %{{y}}<extra></extra>"
        left_title = f"# {left_title_prefix} blocks"

    y_line = ct["pl_rate"].fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=ct.index.tolist(), y=y_bars.tolist(), name=bar_name, hovertemplate=bar_hover)
    fig.add_scatter(
        x=ct.index.tolist(), y=y_line.tolist(), name=line_name, mode="lines+markers",
        hovertemplate=f"Block size %{{x}}<br>{line_name} %{{y:.1f}}%<extra></extra>",
        secondary_y=True,
    )
    left_max = float(y_bars.max()) or 1.0
    right_max = float(y_line.max()) or 1.0

    fig.update_xaxes(
        title_text="Block size (days)",
        type="category", categoryorder="array", categoryarray=cats,
        tickmode="array", tickvals=cats, ticktext=cats, tickangle=0
    )
    fig.update_layout(xaxis=dict(automargin=True))
    fig.update_yaxes(title_text=left_title, range=[0, left_max], secondary_y=False)
    fig.update_yaxes(title_text=line_name, range=[0, right_max], secondary_y=True)
    fig.update_layout(
        title=overall_title, legend_title="", bargap=0.15, margin=dict(l=60, r=40, t=70, b=50),
    )
    return fig

def make_pl_rate_chart_anph_true(df: pd.DataFrame, bars_as_percent: bool = True) -> go.Figure:
    required = {"BLOCK_SIZE", "IF_ANY_PL", "AN_PH"}
    if not required.issubset(df.columns):
        return go.Figure()
    col = df["AN_PH"]
    mask_true = col if pd.api.types.is_bool_dtype(col) else col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})
    d0 = df.loc[mask_true == True].copy()
    if d0.empty:
        return go.Figure()
    return _pl_rate_core(
        d0, bars_as_percent,
        left_title_prefix="PH",
        line_name="% with PL (PH only)",
        overall_title="PL rate vs Block Size — Public holidays only",
    )

def make_pl_rate_chart_anst_true(df: pd.DataFrame, bars_as_percent: bool = True) -> go.Figure:
    required = {"BLOCK_SIZE", "IF_ANY_PL", "AN_ST"}
    if not required.issubset(df.columns):
        return go.Figure()
    col = df["AN_ST"]
    mask_true = col if pd.api.types.is_bool_dtype(col) else col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})
    d0 = df.loc[mask_true == True].copy()
    if d0.empty:
        return go.Figure()
    return _pl_rate_core(
        d0, bars_as_percent,
        left_title_prefix="ST",
        line_name="% with PL (ST only)",
        overall_title="PL rate vs Block Size — Shutdowns only",
    )

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.title("Employee AN Dashboard — PH/ST")

    # Load + clean
    try:
        df = load_csv_upper(DEFAULT_PL_CSV)
        df = coerce_blocksize(df)
        df = coerce_if_any_pl01(df)
        df = coerce_if_any_st01(df)  # harmless if IF_ANY_ST absent
    except Exception as e:
        st.error(f"Couldn't read or parse the CSV.\n\n{e}")
        st.stop()

    # Filters
    filters = sidebar_filter_controls(df)
    df_f = apply_filters(df, filters)

    # HISTOGRAMS (first toggle + PH/ST toggles only)
    render_histograms(df_f)

    # HEATMAPS: ONLY PH==TRUE and ST==TRUE (no "all", no PL-only tab)
    st.subheader("Heatmaps (PH/ST Only)")
    cols = st.columns(2)

    # PH heatmap
    with cols[0]:
        if "AN_PH" in df_f.columns and {"NAME", "BLOCK_SIZE"}.issubset(df_f.columns):
            col = df_f["AN_PH"]
            mask_true = col if pd.api.types.is_bool_dtype(col) else col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})
            df_anph_true = df_f[mask_true == True].copy()
            if not df_anph_true.empty:
                pivot_true, _ = pivot_allblocks(df_anph_true)
                fig_true = make_heatmap(
                    pivot_true,
                    title="Employees by (AN Blocks × Block Size) — Public Holidays Only",
                    y_title="AN blocks per person (per year)",
                )
                st.plotly_chart(fig_true, use_container_width=True)
            else:
                st.info("No rows with AN_PH == TRUE after filters.")
        else:
            st.info("CSV missing 'AN_PH' or required columns for PH heatmap.")

    # ST heatmap
    with cols[1]:
        if "AN_ST" in df_f.columns and {"NAME", "BLOCK_SIZE"}.issubset(df_f.columns):
            col = df_f["AN_ST"]
            mask_true = col if pd.api.types.is_bool_dtype(col) else col.astype("string").str.strip().str.upper().map({"TRUE": True, "FALSE": False})
            df_anst_true = df_f[mask_true == True].copy()
            if not df_anst_true.empty:
                pivot_st, _ = pivot_allblocks(df_anst_true)
                fig_st = make_heatmap(
                    pivot_st,
                    title="Employees by (AN Blocks × Block Size) — Shutdowns Only",
                    y_title="AN blocks per person (per year)",
                )
                st.plotly_chart(fig_st, use_container_width=True)
            else:
                st.info("No rows with AN_ST == TRUE after filters.")
        else:
            st.info("CSV missing 'AN_ST' or required columns for ST heatmap.")

    # DATA SNAPSHOT
    with st.expander("Data snapshot (first 50 rows)"):
        st.dataframe(df_f.head(50))

    # BAR CHARTS: ONLY PH-only and ST-only PL rate charts
    st.markdown("---")
    st.subheader("PL rate vs Block Size — Public holidays only")
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
