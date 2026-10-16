
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# --------------------------------------------------
# File paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT / "dataset" / "processed" / "ml_ready_dataset.csv"
)
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "previous_cpu_1",
    "previous_cpu_2",
    "previous_cpu_3",
    "previous_memory_1",
    "previous_memory_2",
    "previous_memory_3",
    "cpu_usage",
    "memory_usage",
    "time_gap_seconds",
]

CPU_TARGET = "next_cpu"
MEMORY_TARGET = "next_memory"

# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

print("Loading machine-level dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

required = FEATURES + [
    CPU_TARGET,
    MEMORY_TARGET,
    "start_time_seconds",
    "next_time_seconds",
    "machine_id",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

numeric_columns = FEATURES + [
    CPU_TARGET,
    MEMORY_TARGET,
    "start_time_seconds",
    "next_time_seconds",
]

df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric, errors="coerce"
)
df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf], np.nan
)
df = df.dropna(subset=numeric_columns).copy()

df = df[
    df["next_time_seconds"] > df["start_time_seconds"]
].copy()

df = df.sort_values(
    ["start_time_seconds", "machine_id"]
).reset_index(drop=True)

print(f"Usable rows: {len(df):,}")

if len(df) < 20:
    raise ValueError("Too few rows for a meaningful train/test split.")

# --------------------------------------------------
# Chronological split by unique input timestamps
# --------------------------------------------------
# All rows with the same timestamp remain in one partition.
# Training targets must finish before the test cutoff.
# Test targets must occur at or after the cutoff.

unique_times = np.sort(df["start_time_seconds"].unique())

if len(unique_times) < 2:
    raise ValueError("Not enough distinct timestamps to split data.")

cutoff_index = int(len(unique_times) * 0.8)
cutoff_index = min(max(cutoff_index, 1), len(unique_times) - 1)
cutoff_time = unique_times[cutoff_index]

train_df = df[
    (df["start_time_seconds"] < cutoff_time)
    & (df["next_time_seconds"] < cutoff_time)
].copy()

test_df = df[
    (df["start_time_seconds"] >= cutoff_time)
    & (df["next_time_seconds"] >= cutoff_time)
].copy()

print(f"Timestamp cutoff: {cutoff_time}")
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows : {len(test_df):,}")

if len(train_df) < 10 or len(test_df) < 10:
    raise ValueError(
        "Too few rows after the time-safe split. "
        "Inspect the dataset or use a larger source sample."
    )

# Confirm no target crosses the boundary.
assert train_df["next_time_seconds"].max() < cutoff_time
assert test_df["start_time_seconds"].min() >= cutoff_time
assert test_df["next_time_seconds"].min() >= cutoff_time

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_cpu_train = train_df[CPU_TARGET]
y_cpu_test = test_df[CPU_TARGET]

y_memory_train = train_df[MEMORY_TARGET]
y_memory_test = test_df[MEMORY_TARGET]

# --------------------------------------------------
# Train Random Forest models
# --------------------------------------------------

def create_model():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )

print("\nTraining CPU model...")
cpu_model = create_model()
cpu_model.fit(X_train, y_cpu_train)

print("Training memory model...")
memory_model = create_model()
memory_model.fit(X_train, y_memory_train)

cpu_predictions = cpu_model.predict(X_test)
memory_predictions = memory_model.predict(X_test)

# --------------------------------------------------
# Evaluate models and persistence baselines
# --------------------------------------------------

def evaluate_model(actual, predicted, name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")

    return {
        "model": name,
        "test_rows": len(actual),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

metrics = []

metrics.append(
    evaluate_model(
        y_cpu_test, cpu_predictions, "CPU Random Forest"
    )
)
metrics.append(
    evaluate_model(
        y_memory_test, memory_predictions, "Memory Random Forest"
    )
)

# Baseline: next usage is predicted as current usage.
metrics.append(
    evaluate_model(
        y_cpu_test,
        test_df["cpu_usage"],
        "CPU Persistence Baseline",
    )
)
metrics.append(
    evaluate_model(
        y_memory_test,
        test_df["memory_usage"],
        "Memory Persistence Baseline",
    )
)

pd.DataFrame(metrics).to_csv(
    RESULTS_DIR / "model_evaluation.csv",
    index=False,
)

# --------------------------------------------------
# Save models
# --------------------------------------------------

joblib.dump(cpu_model, MODEL_DIR / "cpu_model.pkl")
joblib.dump(memory_model, MODEL_DIR / "memory_model.pkl")

# --------------------------------------------------
# Save test predictions
# --------------------------------------------------

predictions = test_df[
    [
        "machine_id",
        "start_time_seconds",
        "next_time_seconds",
        "cpu_usage",
        "memory_usage",
        "next_cpu",
        "next_memory",
    ]
].copy()

predictions["predicted_cpu"] = cpu_predictions
predictions["predicted_memory"] = memory_predictions

predictions.to_csv(
    RESULTS_DIR / "predictions.csv",
    index=False,
)

print("\n----------------------------------------")
print("TRAINING COMPLETED")
print("----------------------------------------")
print(f"Models:      {MODEL_DIR}")
print(f"Predictions: {RESULTS_DIR / 'predictions.csv'}")
print(f"Metrics:     {RESULTS_DIR / 'model_evaluation.csv'}")
