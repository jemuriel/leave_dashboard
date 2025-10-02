import pandas as pd
import numpy as np

# --- Load & prep ---
# path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\leave_data_v2.csv"
path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\new_data_analysis\Leave_new_data.csv"
# Public holidays table: needs columns STATE and DATE (one row per holiday date)
# Example path (change to your actual file):
holidays_path = r"C:\Users\61432\OneDrive - Pacific National\Leave_data\new_data_analysis\public_holidays.csv"

df = pd.read_csv(path)

# Ensure DATE is proper datetime (keep existing format)
df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y %H:%M', errors='coerce')

# Defensive clean-ups
df['STATE'] = df['STATE'].astype(str).str.strip().str.upper()
df['DATE_ONLY'] = df['DATE'].dt.date  # for date-only joins

# Load holidays and normalise
hol = pd.read_csv(holidays_path, encoding="cp1252")
# Accept flexible columns; must contain STATE and DATE
if 'STATE' not in hol.columns or 'DATE' not in hol.columns:
    raise ValueError("Holidays file must contain columns: STATE, DATE")

hol = hol.copy()
hol['STATE'] = hol['STATE'].astype(str).str.strip().str.upper()
hol['DATE'] = pd.to_datetime(hol['DATE'], errors='coerce')
hol['DATE_ONLY'] = hol['DATE'].dt.date
hol = hol[['STATE', 'DATE_ONLY']].dropna().drop_duplicates()
hol['IS_PUBLIC_HOL'] = True

# Merge flag onto roster rows (row is a public holiday if its DATE_ONLY/STATE pair appears in holidays)
df = df.merge(hol, on=['STATE', 'DATE_ONLY'], how='left')
df['IS_PUBLIC_HOL'] = df['IS_PUBLIC_HOL'].fillna(False)

# Sort
df = df.sort_values(["NAME", "DATE"], kind="mergesort").reset_index(drop=True)

# --- Segment labelling: increment at each WORK within each NAME ---
df["seg_id"] = df.groupby("NAME")["TASK"].transform(lambda s: s.eq("WORK").cumsum())

# Mask: rows strictly between two WORKs (non-WORK rows, and not after the last WORK)
between = df["TASK"].ne("WORK")
last_seg_per_name = df.groupby("NAME")["seg_id"].transform("max")
valid_between = between & (df["seg_id"] < last_seg_per_name)

# Convenience flags
is_AN = df["TASK"].eq("AN")
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

# 4) holiday intersections (booleans) per segment for AN and PL
an_hol_any = (
    df.loc[valid_between & is_AN & df['IS_PUBLIC_HOL']]
      .groupby(["NAME", "seg_id"], sort=False)
      .size()
      .rename("AN_HOL_COUNT")
)
pl_hol_any = (
    df.loc[valid_between & is_PL & df['IS_PUBLIC_HOL']]
      .groupby(["NAME", "seg_id"], sort=False)
      .size()
      .rename("PL_HOL_COUNT")
)

# Combine stats
seg_stats = (
    pd.concat([seg_len, total_AN, total_PL, an_hol_any, pl_hol_any], axis=1)
      .fillna({"TOTAL_AN": 0, "TOTAL_PL": 0, "AN_HOL_COUNT": 0, "PL_HOL_COUNT": 0})
)

# Build output values per (NAME, seg_id)
out_cols = pd.DataFrame(index=seg_stats.index)
out_cols["COUNT_BETWEEN_WORK"] = seg_stats["SEG_LEN"].astype(int)
out_cols["COUNT_BETWEEN_WORK_IF_ANY_AN"] = np.where(seg_stats["TOTAL_AN"] > 0, seg_stats["SEG_LEN"], 0).astype(int)
out_cols["TOTAL_NUMBER_OF_AN_BETWEEN_WORK"] = seg_stats["TOTAL_AN"].astype(int)
out_cols["TOTAL_NUMBER_OF_PL_BETWEEN_WORK"] = seg_stats["TOTAL_PL"].astype(int)

# New: public holiday intersection flags (True if any AN/PL day in the segment is a public holiday)
out_cols["AN_INTERSECTS_PUBLIC_HOL"] = seg_stats["AN_HOL_COUNT"].astype(int).gt(0)
out_cols["PL_INTERSECTS_PUBLIC_HOL"] = seg_stats["PL_HOL_COUNT"].astype(int).gt(0)

# (Optional) If you’d like counts instead of booleans, you already have AN_HOL_COUNT / PL_HOL_COUNT in seg_stats

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
df.to_csv(r"C:\Users\61432\OneDrive - Pacific National\Leave_data\new_data_analysis\leave_analysis_new_data_v3.csv", index=False)
