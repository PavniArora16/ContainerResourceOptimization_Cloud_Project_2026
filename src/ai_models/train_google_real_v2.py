
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path(
    "dataset/processed/google_cluster_real/ml_ready_dataset_v2.csv"
)
OUT = Path("results/google_cluster_real_v2")
MODEL_DIR = OUT / "models"

OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

TARGETS = {
    "cpu": "cpu_usage",
    "memory": "memory_usage",
}

print("Loading V2 dataset...")
df = pd.read_csv(DATA)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + list(TARGETS.values()))
df = df.sort_values("start_seconds").reset_index(drop=True)

# Split by time, never by randomly mixing timestamps.
timestamps = np.sort(df["start_seconds"].unique())

if len(timestamps) < 10:
    raise ValueError("Not enough timestamps for a chronological split.")

cutoff_index = int(len(timestamps) * 0.80)
cutoff_time = timestamps[cutoff_index]

train = df[df["start_seconds"] < cutoff_time].copy()
test = df[df["start_seconds"] >= cutoff_time].copy()

if train.empty or test.empty:
    raise ValueError("Training or test data is empty.")

print(f"Total rows: {len(df):,}")
print(f"Training rows before sampling: {len(train):,}")
print(f"Test rows: {len(test):,}")
print(f"Training period ends before {cutoff_time:.2f} seconds")
print(f"Test period starts at {cutoff_time:.2f} seconds")

# Limit training data to keep memory and runtime manageable.
MAX_TRAIN_ROWS = 250_000

if len(train) > MAX_TRAIN_ROWS:
    train = train.sample(
        n=MAX_TRAIN_ROWS,
        random_state=42,
    )

print(f"Training rows used: {len(train):,}")

X_train = train[FEATURES]
X_test = test[FEATURES]

metrics = []
predictions = pd.DataFrame({
    "start_seconds": test["start_seconds"].to_numpy(),
    "actual_cpu": test["cpu_usage"].to_numpy(),
    "cpu_persistence": test["cpu_lag_1"].to_numpy(),
    "actual_memory": test["memory_usage"].to_numpy(),
    "memory_persistence": test["memory_lag_1"].to_numpy(),
})

for name, target in TARGETS.items():
    print(f"\nTraining {name.upper()} Random Forest...")
    started = time.time()

    y_train = train[target]
    y_test = test[target]

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # Persistence predicts the current value using the last observation.
    baseline = test[
        "cpu_lag_1" if name == "cpu" else "memory_lag_1"
    ].to_numpy()

    for model_name, values in [
        ("RandomForestV2", predicted),
        ("PersistenceBaseline", baseline),
    ]:
        metrics.append({
            "target": name,
            "model": model_name,
            "test_rows": len(y_test),
            "MAE": mean_absolute_error(y_test, values),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, values))),
            "R2": r2_score(y_test, values),
        })

    predictions[f"{name}_random_forest"] = predicted

    model_path = MODEL_DIR / f"{name}_model_v2.pkl"
    joblib.dump(model, model_path)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUT / f"{name}_feature_importance.csv",
        index=False,
    )

    print(f"Saved model: {model_path}")
    print(f"Training time: {time.time() - started:.1f} seconds")

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    OUT / "model_evaluation_v2.csv",
    index=False,
)

predictions.to_csv(
    OUT / "predictions_v2.csv",
    index=False,
)

with open(OUT / "run_info_v2.json", "w", encoding="utf-8") as file:
    json.dump({
        "dataset": str(DATA),
        "total_rows": len(df),
        "training_rows_used": len(train),
        "test_rows": len(test),
        "cutoff_time_seconds": float(cutoff_time),
        "features": FEATURES,
        "targets": TARGETS,
        "training_sample_limit": MAX_TRAIN_ROWS,
    }, file, indent=2)

print("\nV2 MODEL EVALUATION")
print(
    metrics_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nSaved evaluation:", OUT / "model_evaluation_v2.csv")
print("Saved predictions:", OUT / "predictions_v2.csv")
print("Training complete.")
