import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# df has columns: NAME, DEPO, STATE, BLOCK_SIZE (int), IF_ANY_PL (0/1)


def bar_plot():
    # Read safely (avoid DtypeWarning)
    df = pd.read_csv(r"C:\Users\61432\OneDrive - Pacific National\Leave_data\AN_vs_PL.csv",
                     low_memory=False)

    # Normalise column names just in case
    df.columns = [c.strip().upper() for c in df.columns]

    # --- Clean BLOCK_SIZE to numeric
    size = pd.to_numeric(df["BLOCK_SIZE"], errors="coerce")
    # keep only valid, positive sizes
    mask = size.notna() & (size > 0)
    df = df.loc[mask].copy()
    size = size.loc[mask]

    # --- Clean IF_ANY_PL to 0/1
    if "IF_ANY_PL" not in df.columns:
        raise KeyError("Column 'IF_ANY_PL' not found in the data.")

    # map common textual/bool variants to 0/1
    pl_map = {
        "1": 1, "true": 1, "t": 1, "y": 1, "yes": 1, "pl": 1,
        "0": 0, "false": 0, "f": 0, "n": 0, "no": 0, "": 0, "nan": 0
    }
    if df["IF_ANY_PL"].dtype == "O":
        df["IF_ANY_PL"] = (
            df["IF_ANY_PL"].astype(str).str.strip().str.lower().map(pl_map)
        )

    df["IF_ANY_PL"] = pd.to_numeric(df["IF_ANY_PL"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # --- Bucket sizes: 1..15 and >15 (round to nearest day; swap to np.floor if preferred)
    df["SIZE_BUCKET"] = np.where(
        size > 15,
        ">15",
        size.clip(upper=15).round().astype(int).astype(str)
    )

    # --- Aggregate by bucket and PL flag
    ct = (
        df.groupby(["SIZE_BUCKET", "IF_ANY_PL"])
          .size()
          .unstack(fill_value=0)
          .rename(columns={0: "no_pl", 1: "pl"})
    )

    ct["n"] = ct["no_pl"] + ct["pl"]
    total_n = ct["n"].sum()
    ct["pct_total"] = (ct["n"] / total_n * 100).fillna(0)
    ct["pl_rate"] = (ct["pl"] / ct["n"] * 100).fillna(0)

    # --- Order x as 1..15, >15 and drop empty buckets
    cats = [str(i) for i in range(1, 16)] + [">15"]
    ct = ct.reindex(cats).dropna(how="all")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bars = % of total blocks
    fig.add_bar(
        x=ct.index.tolist(),
        y=ct["pct_total"],
        name="% of total blocks",
        hovertemplate="Block size %{x}<br>% of total %{y:.1f}%<extra></extra>",
    )

    # Line = % with PL
    fig.add_scatter(
        x=ct.index.tolist(),
        y=ct["pl_rate"],
        name="% with PL",
        mode="lines+markers",
        hovertemplate="Block size %{x}<br>PL rate %{y:.1f}%<extra></extra>",
        secondary_y=True,
    )

    fig.update_xaxes(
        title_text="Block size (days)",
        categoryorder="array",
        categoryarray=cats,
    )
    # Cap each y-axis at its series max
    left_max = float(ct["pct_total"].max()) if len(ct) else 0.0
    right_max = float(ct["pl_rate"].max()) if len(ct) else 0.0

    # (Optional) avoid a zero-height axis if everything is 0
    left_max = left_max or 1.0
    right_max = right_max or 1.0

    fig.update_yaxes(title_text="% of total blocks", range=[0, left_max], secondary_y=False)
    fig.update_yaxes(title_text="% with PL", range=[0, right_max], secondary_y=True)

    fig.update_layout(
        title="PL rate vs Block Size (bucketed, % of total)",
        legend_title="",
        bargap=0.15,
    )
    fig.show()

bar_plot()


# -------- ---------- HEATMAP -------------------------------------------------
# heatmap_pl_only.py
def heatmap():
    # --- Load your data ---
    csv_path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\AN_vs_PL.csv"
    df = pd.read_csv(csv_path, low_memory=False)

    # --- Normalise columns & types ---
    df.columns = [c.strip().upper() for c in df.columns]

    # Robust coercion of IF_ANY_PL to 0/1
    s = df["IF_ANY_PL"]
    df["IF_ANY_PL"] = (
        pd.to_numeric(s, errors="coerce").gt(0)  # numeric-like -> 1 if >0
          .fillna(s.astype(str).str.strip().str.lower()
                    .isin({"1","1.0","y","yes","true","t","pl"}))
          .astype(int)
    )

    # Ensure BLOCK_SIZE is numeric
    df["BLOCK_SIZE"] = pd.to_numeric(df["BLOCK_SIZE"], errors="coerce")
    df = df.dropna(subset=["BLOCK_SIZE"]).copy()
    df["BLOCK_SIZE"] = df["BLOCK_SIZE"].astype(int)

    # (Optional) filter by DEPO/STATE before building the chart:
    # df = df[df["DEPO"].eq("GRETA - NSWCOL")]
    # df = df[df["STATE"].eq("NSW")]

    # --- Keep only PL==1 records ---
    df_pl = df[df["IF_ANY_PL"] == 1].copy()
    if df_pl.empty:
        raise SystemExit("No rows with IF_ANY_PL == 1 after filtering.")

    # --- Compute # of PL blocks per person (within PL subset) ---
    blocks_per_person_pl = (
        df_pl.groupby("NAME", as_index=False)
             .size()
             .rename(columns={"size": "NUM_PL_BLOCKS"})
    )

    # One row per (NAME, BLOCK_SIZE) among PL==1 rows
    name_block = df_pl.groupby(["NAME", "BLOCK_SIZE"], as_index=False).size().drop(columns="size")

    df_uniq = name_block.merge(blocks_per_person_pl, on="NAME", how="left")

    # --- Buckets ---
    df_uniq["BLOCK_SIZE_BUCKET"] = np.where(df_uniq["BLOCK_SIZE"] <= 20, df_uniq["BLOCK_SIZE"], ">20").astype(str)
    df_uniq["NUM_BLOCKS_BUCKET"] = np.where(df_uniq["NUM_PL_BLOCKS"] <= 10, df_uniq["NUM_PL_BLOCKS"], ">10").astype(str)

    # --- Pivot to counts ---
    x_order = [str(i) for i in range(1, 21)] + [">20"]
    y_order = [str(i) for i in range(1, 11)] + [">10"]

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

    # --- Plotly heatmap (Viridis palette preserved) ---
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            colorbar=dict(title="Employees"),
            zmin=0,
            hovertemplate="Block size: %{x}<br>PL blocks/person: %{y}<br>Employees: %{z}<extra></extra>",
        )
    )

    fig.update_yaxes(autorange="reversed", title="PL blocks per person (per year)")
    fig.update_xaxes(title="Block size (days)")
    fig.update_layout(
        title="Employees with PL by (PL Blocks per Person × Block Size)",
        margin=dict(l=60, r=40, t=70, b=50),
    )

    fig.show()

