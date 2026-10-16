
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# File paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "smart_factory_workload_sample.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "ml_ready_dataset.csv"
)

# --------------------------------------------------
# Load source data
# --------------------------------------------------

print("Loading Google Cluster workload sample...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Input rows: {len(df):,}")

required = [
    "start_time",
    "machine_id",
    "cpu_usage",
    "memory_usage",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# --------------------------------------------------
# Clean required columns
# --------------------------------------------------

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required).copy()

# Google Cluster timestamps are in microseconds.
df["start_time_seconds"] = df["start_time"] / 1_000_000

# --------------------------------------------------
# Aggregate task usage per machine and timestamp
# --------------------------------------------------
# Multiple tasks can have records at the same machine/time.
# Sum their normalized CPU and memory usage to estimate
# the total recorded load for that machine and timestamp.

machine_data = (
    df.groupby(
        ["machine_id", "start_time", "start_time_seconds"],
        as_index=False,
    )
    .agg(
        cpu_usage=("cpu_usage", "sum"),
        memory_usage=("memory_usage", "sum"),
        task_count=("machine_id", "size"),
    )
)

# --------------------------------------------------
# Sort observations for each machine
# --------------------------------------------------

machine_data = machine_data.sort_values(
    ["machine_id", "start_time_seconds"]
).reset_index(drop=True)

grouped = machine_data.groupby("machine_id", sort=False)

# --------------------------------------------------
# Create previous-observation features
# --------------------------------------------------

for lag in [1, 2, 3]:
    machine_data[f"previous_cpu_{lag}"] = (
        grouped["cpu_usage"].shift(lag)
    )

    machine_data[f"previous_memory_{lag}"] = (
        grouped["memory_usage"].shift(lag)
    )

# Time elapsed since the previous observation.
machine_data["time_gap_seconds"] = (
    grouped["start_time_seconds"].diff()
)

# --------------------------------------------------
# Create next-observation targets
# --------------------------------------------------

machine_data["next_cpu"] = (
    grouped["cpu_usage"].shift(-1)
)

machine_data["next_memory"] = (
    grouped["memory_usage"].shift(-1)
)

machine_data["next_time_seconds"] = (
    grouped["start_time_seconds"].shift(-1)
)

# --------------------------------------------------
# Keep valid observations only
# --------------------------------------------------

feature_and_target_columns = [
    "previous_cpu_1",
    "previous_cpu_2",
    "previous_cpu_3",
    "previous_memory_1",
    "previous_memory_2",
    "previous_memory_3",
    "cpu_usage",
    "memory_usage",
    "time_gap_seconds",
    "next_cpu",
    "next_memory",
    "next_time_seconds",
]

machine_data[feature_and_target_columns] = (
    machine_data[feature_and_target_columns]
    .replace([np.inf, -np.inf], np.nan)
)

ml_df = machine_data.dropna(
    subset=feature_and_target_columns
).copy()

# Only keep targets at strictly later timestamps.
ml_df = ml_df[
    ml_df["next_time_seconds"] > ml_df["start_time_seconds"]
].copy()

# Final chronological ordering.
ml_df = ml_df.sort_values(
    ["start_time_seconds", "machine_id"]
).reset_index(drop=True)

# --------------------------------------------------
# Save ML-ready dataset
# --------------------------------------------------

output_columns = [
    "machine_id",
    "start_time_seconds",
    "next_time_seconds",
    "task_count",
    "cpu_usage",
    "memory_usage",
    "previous_cpu_1",
    "previous_cpu_2",
    "previous_cpu_3",
    "previous_memory_1",
    "previous_memory_2",
    "previous_memory_3",
    "time_gap_seconds",
    "next_cpu",
    "next_memory",
]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

ml_df[output_columns].to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nPreprocessing completed.")
print(f"Input task records: {len(df):,}")
print(f"Machine-time records: {len(machine_data):,}")
print(f"ML-ready rows: {len(ml_df):,}")
print(f"Machines represented: {ml_df['machine_id'].nunique():,}")
print(f"Zero-gap rows: {(ml_df['time_gap_seconds'] == 0).sum():,}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nPreview:")
print(ml_df[output_columns].head().to_string(index=False))
