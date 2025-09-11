import pandas as pd
import numpy as np

# --- Load & prep ---
# path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\leave_data_v2.csv"
path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\leave_data_latest.csv"
df = pd.read_csv(path)

# Build DATE from YEAR/MONTH/DAY and sort
# df["DATE"] = pd.to_datetime(
#     dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"]),
#     errors="coerce"
# )

df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)

df = df.sort_values(["NAME", "DATE"], kind="mergesort").reset_index(drop=True)

# --- Segment labelling: increment at each WORK within each NAME ---
df["seg_id"] = df.groupby("NAME")["TASK"].transform(lambda s: s.eq("WORK").cumsum())

# Mask: rows strictly between two WORKs (non-WORK rows, and not after the last WORK)
between = df["TASK"].ne("WORK")
last_seg_per_name = df.groupby("NAME")["seg_id"].transform("max")
valid_between = between & (df["seg_id"] < last_seg_per_name)

# Convenience flags
# is_AN = df["TASK"].eq("AN")
is_AN = df["TASK_CTT_CODE"].eq("AN")
is_PL = df["TASK"].eq("PL")

# --- Per-segment stats over valid-between rows ---
# 1) segment length
seg_len = (
    df.loc[valid_between]
      .groupby(["NAME", "seg_id"], sort=False)
      .size()
      .rename("SEG_LEN")
)

# 2) total number of AN rows in each segment
total_AN = (
    df.loc[valid_between & is_AN]
      .groupby(["NAME", "seg_id"], sort=False)
      .size()
      .rename("TOTAL_AN")
)

# 3) total number of PL rows in each segment
total_PL = (
    df.loc[valid_between & is_PL]
      .groupby(["NAME", "seg_id"], sort=False)
      .size()
      .rename("TOTAL_PL")
)

# Combine stats
seg_stats = (
    pd.concat([seg_len, total_AN, total_PL], axis=1)
      .fillna({"TOTAL_AN": 0, "TOTAL_PL": 0})
)

# Build output values per (NAME, seg_id)
out_cols = pd.DataFrame(index=seg_stats.index)
out_cols["COUNT_BETWEEN_WORK"] = seg_stats["SEG_LEN"].astype(int)
out_cols["COUNT_BETWEEN_WORK_IF_ANY_AN"] = np.where(seg_stats["TOTAL_AN"] > 0, seg_stats["SEG_LEN"], 0).astype(int)
out_cols["TOTAL_NUMBER_OF_AN_BETWEEN_WORK"] = seg_stats["TOTAL_AN"].astype(int)
out_cols["TOTAL_NUMBER_OF_PL_BETWEEN_WORK"] = seg_stats["TOTAL_PL"].astype(int)

# --- Mark only the first row after WORK for each (NAME, seg_id) ---
first_idx_per_seg = (
    df.loc[valid_between]
      .groupby(["NAME", "seg_id"], sort=False)["DATE"]
      .idxmin()
)

# Prepare destination columns
for col in out_cols.columns:
    df[col] = np.nan

# Join values onto those first rows
first_keys = df.loc[first_idx_per_seg, ["NAME", "seg_id"]]
first_values = first_keys.join(out_cols, on=["NAME", "seg_id"])

df.loc[first_idx_per_seg, out_cols.columns] = first_values[out_cols.columns].to_numpy()

# Clean up
df = df.drop(columns=["seg_id"])

# Save
df.to_csv(r"C:\Users\61432\OneDrive - Pacific National\Leave_data\leave_analysis_latest_AN.csv", index=False)
