# streamlit_app_basic.py
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
st.set_page_config(page_title="Leave Granularity — Basic", layout="wide")

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# st.write(PROJECT_ROOT)
CSV_FOLDER = PROJECT_ROOT / "csv"

DEFAULT_PL_CSV = CSV_FOLDER / "an_pl_ph_data.csv"  # file can still have PH/ST cols; we just ignore them
SENTINEL_YEARS = 125  # special value

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

# =============================================================================
# SIDEBAR FILTERS (no PH/ST)
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

        if sel_depos is not None and "DEPO" in base.columns:
            base = base[base["DEPO"].isin(sel_depos)]

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
                sel_years_range = None
                include_125 = True

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
# HISTOGRAMS (no PH/ST toggles)
# =============================================================================
def render_histograms(df_f: pd.DataFrame):
    st.subheader("Distributions")
    df_hist = df_f.copy()

    if "GENDER" in df_hist.columns:
        g = df_hist["GENDER"].astype("string").str.strip().str.lower()
        df_hist["GENDER_N"] = g.map({"m": "Male", "male": "Male", "f": "Female", "female": "Female"}).astype("string")
    else:
        df_hist["GENDER_N"] = pd.Series(pd.NA, index=df_hist.index, dtype="string")

    hist_pct = st.toggle("Show histogram y-axis as %", value=False, key="hist_pct")
    gender_split = st.toggle("Split histograms by gender (M/F)", value=False, key="hist_gender_split")
    facet_by_grade = st.toggle("Small multiples by JOB_TYPE", value=False, key="facet_by_grade")

    COLORS = {"Male": "#1f77b4", "Female": "#e377c2"}

    # Facet path (by JOB_TYPE)
    if facet_by_grade and "JOB_TYPE" in df_hist.columns and not df_hist.empty:
        # Block size
        cats = [str(i) for i in range(1, 21)] + [">20"]
        recs = []
        for grade in sorted(df_hist["JOB_TYPE"].dropna().unique().tolist()):
            d_g = df_hist[df_hist["JOB_TYPE"] == grade]
            s = pd.to_numeric(d_g["BLOCK_SIZE"], errors="coerce")
            s = s[s.notna() & (s > 0)]
            if s.empty:
                continue
            bucket = np.where(s > 20, ">20", s.clip(upper=20).astype(int).astype(str))
            if gender_split:
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
                df_bsize, x="bucket", y="value", facet_col="JOB_TYPE", facet_col_wrap=4,
                color="group" if gender_split else None,
                category_orders={"bucket": cats},
                labels={"bucket": "Block size (1–20, >20)", "value": "% of blocks" if hist_pct else "Count"},
                title="Distribution of Block Size — by JOB_TYPE",
            )
            fig_grid1.for_each_yaxis(lambda a: a.update(matches="y"))
            if gender_split:
                fig_grid1.update_traces(opacity=0.6)
                fig_grid1.update_layout(barmode="overlay")
                fig_grid1.for_each_trace(lambda t: t.update(marker_color=COLORS.get(t.name, None)))
            fig_grid1.update_layout(bargap=0.05, margin=dict(l=60, r=40, t=60, b=50))
            if hist_pct and not df_bsize.empty:
                ymax = min(100.0, max(1.0, float(df_bsize["value"].max()) * 1.08))
                fig_grid1.layout.yaxis.update(range=[0, ymax])
            st.plotly_chart(fig_grid1, use_container_width=True)

        # Blocks per person
        grade_per_name = (
            df_hist.groupby("NAME")["JOB_TYPE"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iloc[0] if s.dropna().size else pd.NA))
        )
        blocks_per_person = (
            df_hist.groupby("NAME", as_index=False)
            .size()
            .rename(columns={"size": "NUM_BLOCKS"})
        ).merge(grade_per_name.reset_index(), on="NAME", how="left")

        recs2 = []
        if not blocks_per_person.empty:
            if gender_split:
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
                if gender_split:
                    for grp_name, d_grp in d.groupby("GROUP"):
                        vc = d_grp["NUM_BLOCKS"].value_counts().sort_index()
                        if max_nb > 0:
                            vc = vc.reindex(range(1, max_nb + 1), fill_value=0)
                        y = (vc / vc.sum() * 100.0) if (hist_pct and vc.sum()) else vc
                        for k, v in y.items():
                            recs2.append(
                                {"JOB_TYPE": str(grade), "group": str(grp_name), "num_blocks": int(k), "value": float(v)}
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
                df_numblocks, x="num_blocks", y="value", facet_col="JOB_TYPE", facet_col_wrap=4,
                color="group" if gender_split else None,
                labels={"num_blocks": "AN blocks per person (year)", "value": "% of employees" if hist_pct else "Employees"},
                title="Distribution of AN Blocks per Person — by JOB_TYPE",
            )
            fig_grid2.for_each_yaxis(lambda a: a.update(matches="y"))
            if gender_split:
                fig_grid2.update_traces(opacity=0.6)
                fig_grid2.update_layout(barmode="overlay")
                fig_grid2.for_each_trace(lambda t: t.update(marker_color=COLORS.get(t.name, None)))
            fig_grid2.update_layout(bargap=0.05, margin=dict(l=60, r=40, t=60, b=50))
            if not df_numblocks.empty:
                fig_grid2.update_xaxes(tickmode="linear", dtick=1)
            if hist_pct and not df_numblocks.empty:
                ymax2 = min(100.0, max(1.0, float(df_numblocks["value"].max()) * 1.08))
                fig_grid2.layout.yaxis.update(range=[0, ymax2])
            st.plotly_chart(fig_grid2, use_container_width=True)
            return

    # Non-faceted pair
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
            fig_h1.update_layout(
                title="Distribution of Block Size (days)",
                xaxis_title="Block size (1–20, >20)",
                yaxis_title=y_title,
                bargap=0.05,
                margin=dict(l=60, r=40, t=60, b=50),
                xaxis=dict(type="category"),
            )
            st.plotly_chart(fig_h1, use_container_width=True)

        # Right: blocks per person
        with c2:
            max_nb = int(blocks_per_person["NUM_BLOCKS"].max()) if not blocks_per_person.empty else 0
            fig_h2 = go.Figure(
                go.Histogram(
                    x=blocks_per_person["NUM_BLOCKS"],
                    xbins=dict(start=1, end=max_nb + 1, size=1) if max_nb > 0 else None,
                    histnorm="percent" if hist_pct else None,
                    name="All employees",
                )
            )
            fig_h2.update_layout(
                title="Distribution of AN Blocks per Person",
                xaxis_title="AN blocks per person (year)",
                yaxis_title="% of employees" if hist_pct else "Employees",
                bargap=0.05,
                margin=dict(l=60, r=40, t=60, b=50),
            )
            if max_nb > 0:
                tickvals = [i + 0.5 for i in range(1, max_nb + 1)]
                ticktext = list(range(1, max_nb + 1))
                fig_h2.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            st.plotly_chart(fig_h2, use_container_width=True)
    else:
        st.info("CSV must include 'NAME' and 'BLOCK_SIZE' to draw the histograms.")

# =============================================================================
# PL RATE (overall only — no PH/ST)
# =============================================================================
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

    cats = [str(i) for i in range(1, 16)] + [">15"]
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
    fig.add_bar(x=ct.index.tolist(), y=y_bars.tolist(), name=bar_name, hovertemplate=bar_hover)
    fig.add_scatter(
        x=ct.index.tolist(),
        y=y_line.tolist(),
        name="% with PL",
        mode="lines+markers",
        hovertemplate="Block size %{x}<br>PL rate %{y:.1f}%<extra></extra>",
        secondary_y=True,
    )
    left_max = float(y_bars.max()) or 1.0
    right_max = float(y_line.max()) or 1.0

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

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.title("Employee AN Dashboard — Basic")

    # Load + clean
    try:
        df = load_csv_upper(DEFAULT_PL_CSV)
        df = coerce_blocksize(df)
        df = coerce_if_any_pl01(df)
    except Exception as e:
        st.error(f"Couldn't read or parse the CSV.\n\n{e}")
        st.stop()

    # Filters
    filters = sidebar_filter_controls(df)
    df_f = apply_filters(df, filters)

    # Histograms
    render_histograms(df_f)

    # Heatmaps (ONLY: all blocks + PL-only; NO PH/ST heatmaps)
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

    # PL rate (overall only)
    st.markdown("---")
    st.subheader("PL rate vs Block Size")
    bars_as_percent = st.toggle("Show bars as % of total", value=True)
    if {"BLOCK_SIZE", "IF_ANY_PL"}.issubset(df_f.columns) and not df_f.empty:
        fig_rate = make_pl_rate_chart(df_f, bars_as_percent=bars_as_percent)
        if fig_rate.data:
            st.plotly_chart(fig_rate, use_container_width=True)
        else:
            st.info("Not enough data to render the PL rate chart.")
    else:
        st.info("CSV must include 'BLOCK_SIZE' and 'IF_ANY_PL' to render the PL rate chart.")

if __name__ == "__main__":
    main()
