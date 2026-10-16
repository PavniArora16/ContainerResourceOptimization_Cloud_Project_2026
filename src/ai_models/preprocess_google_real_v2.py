
from pathlib import Path

import pandas as pd
import numpy as np

# Genuine Google Cluster task_usage shard
SOURCE = Path(
    r"D:\GoogleClusterData\task_usage\part-00000-of-00500.csv.gz"
)

OUTPUT = Path(
    "dataset/processed/google_cluster_real/ml_ready_dataset_v2.csv"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "start_time",
    "end_time",
    "job_id",
    "task_index",
    "machine_id",
    "cpu_usage",
    "memory_usage",
]

print("Reading genuine Google Cluster data in chunks...")

chunks = []

for number, chunk in enumerate(
    pd.read_csv(
        SOURCE,
        header=None,
        usecols=list(range(7)),
        names=COLUMNS,
        compression="gzip",
        chunksize=200_000,
    ),
    start=1,
):
    chunk = chunk.dropna(subset=COLUMNS)

    chunk = chunk[
        (chunk["end_time"] > chunk["start_time"])
        & (chunk["cpu_usage"] >= 0)
        & (chunk["memory_usage"] >= 0)
    ]

    # Keep identifiers as integers and usage values compact.
    for col in ["start_time", "end_time", "job_id",
                "task_index", "machine_id"]:
        chunk[col] = chunk[col].astype("int64")

    chunk["cpu_usage"] = chunk["cpu_usage"].astype("float32")
    chunk["memory_usage"] = chunk["memory_usage"].astype("float32")

    chunks.append(chunk)
    print(f"Read chunk {number}")

df = pd.concat(chunks, ignore_index=True)
del chunks

print("Valid observations:", len(df))

# Sort and remove duplicate task intervals.
df = df.sort_values(
    ["job_id", "task_index", "start_time", "end_time"]
)

df = df.drop_duplicates(
    subset=["job_id", "task_index", "start_time", "end_time"]
).copy()

task_keys = ["job_id", "task_index"]
groups = df.groupby(task_keys, sort=False)

# Previous observations for each task.
for lag in (1, 2, 3):
    df[f"cpu_lag_{lag}"] = groups["cpu_usage"].shift(lag)
    df[f"memory_lag_{lag}"] = groups["memory_usage"].shift(lag)

df["previous_end_time"] = groups["end_time"].shift(1)

# Require the immediately previous interval to finish before
# the current interval begins.
df["gap_seconds"] = (
    df["start_time"] - df["previous_end_time"]
) / 1_000_000

df = df[
    df["previous_end_time"].notna()
    & (df["gap_seconds"] >= 0)
    & (df["gap_seconds"] <= 3600)
].copy()

# Rolling means from the previous observations only.
# Shift before rolling to avoid including the target observation.
groups = df.groupby(task_keys, sort=False)

df["cpu_rolling_mean_3"] = groups["cpu_usage"].transform(
    lambda s: s.shift(1).rolling(3, min_periods=3).mean()
)

df["memory_rolling_mean_3"] = groups["memory_usage"].transform(
    lambda s: s.shift(1).rolling(3, min_periods=3).mean()
)

# Remove rows without three prior observations so all rows
# have the same set of historical features.
features = [
    "start_time",
    "gap_seconds",
    "cpu_lag_1",
    "cpu_lag_2",
    "cpu_lag_3",
    "memory_lag_1",
    "memory_lag_2",
    "memory_lag_3",
    "cpu_rolling_mean_3",
    "memory_rolling_mean_3",
]

targets = ["cpu_usage", "memory_usage"]

df = df.dropna(subset=features + targets).copy()

# Use seconds for easier interpretation.
# Preserve actual interval timestamps in seconds.
df["start_seconds"] = df["start_time"] / 1_000_000
df["end_seconds"] = df["end_time"] / 1_000_000

final_columns = [
    "start_seconds",
    "end_seconds",
    "machine_id",
    "gap_seconds",
    "job_id",
    "task_index",
    "cpu_lag_1",
    "cpu_lag_2",
    "cpu_lag_3",
    "memory_lag_1",
    "memory_lag_2",
    "memory_lag_3",
    "cpu_rolling_mean_3",
    "memory_rolling_mean_3",
    "cpu_usage",
    "memory_usage",
]

df = df[final_columns].sort_values(
    ["start_seconds", "job_id", "task_index"]
).reset_index(drop=True)

df.to_csv(OUTPUT, index=False)

print("\nV2 preprocessing complete!")
print("Rows:", len(df))
print("Distinct timestamps:", df["start_seconds"].nunique())
print("Earliest time:", df["start_seconds"].min())
print("Latest time:", df["start_seconds"].max())
print("Output:", OUTPUT)
