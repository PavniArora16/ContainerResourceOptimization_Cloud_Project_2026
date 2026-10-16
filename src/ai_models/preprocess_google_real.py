
from pathlib import Path
import pandas as pd
import numpy as np

# Genuine Google Cluster 2011-2 task_usage shard
SOURCE = Path(
    r"D:\GoogleClusterData\task_usage\part-00000-of-00500.csv.gz"
)

OUTPUT = Path(
    "dataset/processed/google_cluster_real/ml_ready_dataset.csv"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Read only the columns needed for task-level prediction.
COLUMNS = [
    "start_time",
    "end_time",
    "job_id",
    "task_index",
    "machine_id",
    "cpu_usage",
    "memory_usage",
]

print("Reading genuine Google Cluster data...")

chunks = []

for chunk_number, chunk in enumerate(
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
    chunk = chunk.dropna(
        subset=[
            "start_time",
            "end_time",
            "job_id",
            "task_index",
            "machine_id",
            "cpu_usage",
            "memory_usage",
        ]
    )

    chunk = chunk[
        (chunk["end_time"] > chunk["start_time"])
        & (chunk["cpu_usage"] >= 0)
        & (chunk["memory_usage"] >= 0)
    ]

    # Reduce memory usage
    chunk["cpu_usage"] = chunk["cpu_usage"].astype("float32")
    chunk["memory_usage"] = chunk["memory_usage"].astype("float32")

    chunks.append(chunk)

    print(f"Processed chunk {chunk_number}")

df = pd.concat(chunks, ignore_index=True)
del chunks

print("Valid raw observations:", len(df))

# Sort each task's observations chronologically.
df = df.sort_values(
    ["job_id", "task_index", "start_time", "end_time"]
)

# Remove duplicate measurements for the same task and interval.
df = df.drop_duplicates(
    subset=[
        "job_id",
        "task_index",
        "start_time",
        "end_time",
    ]
)

# Each task is identified by its job ID and task index.
task_groups = df.groupby(
    ["job_id", "task_index"],
    sort=False
)

# Use the previous observation to predict the current observation.
df["previous_cpu"] = task_groups["cpu_usage"].shift(1)
df["previous_memory"] = task_groups["memory_usage"].shift(1)
df["previous_end_time"] = task_groups["end_time"].shift(1)

# Only retain consecutive observations that do not overlap.
df = df[
    df["previous_end_time"].notna()
    & (df["previous_end_time"] <= df["start_time"])
].copy()

# Historical features, available before the target interval.
df["gap_seconds"] = (
    df["start_time"] - df["previous_end_time"]
) / 1_000_000

df["start_seconds"] = df["start_time"] / 1_000_000

# Restrict extreme gaps for this first experiment.
df = df[
    (df["gap_seconds"] >= 0)
    & (df["gap_seconds"] <= 3600)
]

# Features and targets for the ML models.
final_columns = [
    "start_seconds",
    "gap_seconds",
    "previous_cpu",
    "previous_memory",
    "cpu_usage",
    "memory_usage",
]

df = df[final_columns].dropna()

df = df.sort_values("start_seconds").reset_index(drop=True)

df.to_csv(OUTPUT, index=False)

print("\nPreprocessing complete!")
print("ML-ready rows:", len(df))
print("Unique prediction times:", df["start_seconds"].nunique())
print("Earliest prediction time:", df["start_seconds"].min())
print("Latest prediction time:", df["start_seconds"].max())
print("Saved to:", OUTPUT)
