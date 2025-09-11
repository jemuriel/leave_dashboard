import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Example: df_blocks must have NAME, YEAR, BLOCK_SIZE ---
# df_blocks = pd.DataFrame({
#     "NAME": ["A","A","B","B","C","C","C"],
#     "YEAR": [2025]*7,
#     "BLOCK_SIZE": [1, 12, 3, 7, 2, 15, 1]
# })

df_blocks = pd.read_csv(r"C:\Users\61432\OneDrive - Pacific National\Leave_data\granularity_data.csv")

# 1) blocks per person-year
blocks_per_person = (
    df_blocks.groupby("NAME", as_index=False)
             .size()
             .rename(columns={"size": "NUM_BLOCKS"})
)

# 2) unique block sizes per person-year
df_uniq = (
    df_blocks.drop_duplicates(["NAME", "BLOCK_SIZE"])
             .merge(blocks_per_person, on="NAME", how="left")
)

def heatmap():
    # 3) bucket sizes and counts
    df_uniq["BLOCK_SIZE_BUCKET"] = df_uniq["BLOCK_SIZE"].where(df_uniq["BLOCK_SIZE"] <= 20, ">20")
    df_uniq["NUM_BLOCKS_BUCKET"] = df_uniq["NUM_BLOCKS"].where(df_uniq["NUM_BLOCKS"] <= 10, ">10")

    # 4) build frequency table
    heat = (
        df_uniq.groupby(["NUM_BLOCKS_BUCKET", "BLOCK_SIZE_BUCKET"])
        .size()
        .reset_index(name="COUNT")
    )

    # enforce axis order
    x_order = list(range(1, 21)) + [">20"]
    y_order = list(range(1, 11)) + [">10"]

    # pivot
    pivot = heat.pivot(index="NUM_BLOCKS_BUCKET", columns="BLOCK_SIZE_BUCKET", values="COUNT").fillna(0)
    pivot = pivot.reindex(index=y_order, columns=x_order, fill_value=0)

    # 5) plot with matplotlib
    plt.figure(figsize=(10, 7))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")

    plt.title("Employees by (Number of AN Blocks, Block Size)")
    plt.xlabel("Block size (days)")
    plt.ylabel("AN blocks per person (per year)")

    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)

    cbar = plt.colorbar(im)
    cbar.set_label("Employees")

    plt.show()

heatmap()
