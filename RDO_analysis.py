import pandas as pd
import numpy as np

def add_rdo_block_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two columns per (NAME, DEPO):
      - RDO_BLOCK_SIZE: size of each consecutive RDO block (calendar-day adjacency),
        populated only on the first row of the RDO block; NaN elsewhere.
      - RDO_PL_BLOCK_SIZE: same size but only for RDO blocks that are immediately
        preceded or followed (by calendar day) by a PL; populated only on the first
        row of that RDO block; NaN otherwise.
    Expected columns: YEAR, MONTH, DAY, NAME, DEPO, TYPE.
    """
    df = df.copy()

    # Normalise TYPE labels
    t = df["TYPE"].astype(str).str.strip().str.upper()
    t = t.replace({
        "RDO OR ANNUAL LEAVE": "RDO",
        "ANNUAL LEAVE": "RDO",
        "PERSONAL LEAVE": "PL"
    })
    df["TYPE_N"] = df["TYPE"]

    # Build a proper DATE and sort
    df["DATE"] = pd.to_datetime(
        dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"]),
        errors="coerce"
    )
    df = df.sort_values(["NAME", "DEPO", "DATE"]).reset_index(drop=True)

    def per_person(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("DATE").copy()
        is_rdo = g["TYPE_N"].eq("RDO")

        # prev/next helpers within the group
        prev_date = g["DATE"].shift()
        next_date = g["DATE"].shift(-1)
        prev_type = g["TYPE_N"].shift()
        next_type = g["TYPE_N"].shift(-1)

        # Calendar-adjacent starts/ends of RDO runs
        start_rdo = is_rdo & (
            (~is_rdo.shift(fill_value=False)) |
            ((g["DATE"] - prev_date).dt.days != 1)
        )
        end_rdo = is_rdo & (
            (~is_rdo.shift(-1, fill_value=False)) |
            ((next_date - g["DATE"]).dt.days != 1)
        )

        # Raw id for each RDO block, only on RDO rows
        rdo_block_id = start_rdo.cumsum().where(is_rdo)

        # Compute size for each block (count of RDO rows in that block)
        block_sizes = rdo_block_id.value_counts().to_dict()

        # Identify which blocks touch PL on either side by calendar day
        touches_prev_pl = start_rdo & prev_type.eq("PL") & ((g["DATE"] - prev_date).dt.days.eq(1))
        touches_next_pl = end_rdo   & next_type.eq("PL") & ((next_date - g["DATE"]).dt.days.eq(1))

        qualifying_blocks = set(rdo_block_id[touches_prev_pl].dropna().unique()) \
                            | set(rdo_block_id[touches_next_pl].dropna().unique())

        # Prepare output columns as NaN
        g["RDO_BLOCK_SIZE"] = np.nan
        g["RDO_PL_BLOCK_SIZE"] = np.nan

        # Fill only at the first row of each RDO block
        for bid in rdo_block_id[start_rdo].dropna().unique():
            bid = float(bid)  # keys are floats because of NaN in Series
            block_size = int(block_sizes.get(bid, 0))

            # first row index for this block
            idx_first = g.index[(rdo_block_id == bid) & start_rdo].tolist()[0]

            # Always write the RDO block size at the start row
            g.loc[idx_first, "RDO_BLOCK_SIZE"] = block_size

            # Write the PL-touching block size only if qualifying
            if bid in qualifying_blocks:
                g.loc[idx_first, "RDO_PL_BLOCK_SIZE"] = block_size

        return g

    out = df.groupby(["NAME", "DEPO"], group_keys=False).apply(per_person)

    # Return to database order if you like
    out = out.sort_values(["YEAR", "MONTH", "DAY", "NAME", "DEPO"]).reset_index(drop=True)
    return out

# --- Example usage ---
path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\leave_data_v3.csv"
data = pd.read_csv(path)
out = add_rdo_block_sizes(data)
out.to_csv(r"C:\Users\61432\OneDrive - Pacific National\Leave_data\leave_with_rdo_blocks.csv", index=False)
