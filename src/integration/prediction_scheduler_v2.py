
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.scheduler.priority_manager import assign_priority
from src.scheduler.resource_allocator import allocate_resources

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "dataset" / "processed"
    / "google_cluster_real" / "ml_ready_dataset_v2.csv"
)
MODEL_DIR = ROOT / "results" / "google_cluster_real_v2" / "models"
OUTPUT_DIR = ROOT / "results" / "google_cluster_real_v2"
OUTPUT_FILE = OUTPUT_DIR / "resource_allocation_v2.csv"

# Normalized experimental capacities, not physical cores or GB.
TOTAL_CPU = 1.0
TOTAL_MEMORY = 1.0
MAX_TASKS = 1000

FEATURES = [
    "start_seconds",
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


def main():
    print("=" * 55)
    print("GOOGLE CLUSTER V2: PREDICTION TO SCHEDULER")
    print("=" * 55)

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + [
        "job_id",
        "task_index",
        "machine_id",
        "end_seconds",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cpu_model = joblib.load(MODEL_DIR / "cpu_model_v2.pkl")
    memory_model = joblib.load(MODEL_DIR / "memory_model_v2.pkl")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).copy()
    df = df[df["end_seconds"] > df["start_seconds"]].copy()

    # Select evenly spaced rows across the complete available timeline.
    df = df.sort_values(
        ["start_seconds", "job_id", "task_index"]
    ).reset_index(drop=True)

    count = min(MAX_TASKS, len(df))
    positions = np.linspace(0, len(df) - 1, num=count, dtype=int)
    df = df.iloc[positions].copy()

    print(f"Rows selected for simulation: {len(df):,}")
    print(
        f"Selected time range: {df['start_seconds'].min()} "
        f"to {df['start_seconds'].max()} seconds"
    )
    print(
        "Distinct selected start times:",
        df["start_seconds"].nunique(),
    )

    X = df[FEATURES]
    df["predicted_cpu"] = np.maximum(cpu_model.predict(X), 0)
    df["predicted_memory"] = np.maximum(
        memory_model.predict(X), 0
    )

    # Distinguish separate recorded intervals of the same task.
    df["task_id"] = (
        df["job_id"].astype(str) + "_"
        + df["task_index"].astype(str) + "_"
        + df["start_seconds"].astype(str) + "_"
        + df["end_seconds"].astype(str)
    )

    workloads = pd.DataFrame({
        "task_id": df["task_id"],
        "priority": [
            int(assign_priority(float(cpu) + float(memory)))
            for cpu, memory in zip(
                df["predicted_cpu"], df["predicted_memory"]
            )
        ],
        "predicted_cpu": df["predicted_cpu"],
        "predicted_memory": df["predicted_memory"],
        "start_time_seconds": df["start_seconds"],
        "end_time_seconds": df["end_seconds"],
    })

    print("\nRunning resource allocator...")
    allocations = allocate_resources(
        workloads,
        total_cpu=TOTAL_CPU,
        total_memory=TOTAL_MEMORY,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    allocations.to_csv(OUTPUT_FILE, index=False)

    statuses = allocations["allocation_status"].value_counts()
    print("\n" + "=" * 55)
    print("ALLOCATION SUMMARY")
    print("=" * 55)
    print(f"Tasks simulated: {len(allocations):,}")
    print(f"Fully allocated: {statuses.get('FULL', 0):,}")
    print(f"Partially allocated: {statuses.get('PARTIAL', 0):,}")
    print(f"Not allocated: {statuses.get('NOT_ALLOCATED', 0):,}")
    print(f"Normalized CPU capacity: {TOTAL_CPU}")
    print(f"Normalized memory capacity: {TOTAL_MEMORY}")
    print(f"Saved results: {OUTPUT_FILE}")

    print("\nFirst 10 allocation results:")
    print(allocations.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
