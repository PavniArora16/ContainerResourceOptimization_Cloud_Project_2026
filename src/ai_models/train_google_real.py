
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA = Path("dataset/processed/google_cluster_real/ml_ready_dataset.csv")
OUT = Path("results/google_cluster_real")
MODEL_DIR = OUT / "models"

OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "start_seconds",
    "gap_seconds",
    "previous_cpu",
    "previous_memory",
]

TARGETS = {
    "cpu": "cpu_usage",
    "memory": "memory_usage",
}

print("Loading prepared Google Cluster data...")
df = pd.read_csv(DATA)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + list(TARGETS.values()))
df = df.sort_values("start_seconds").reset_index(drop=True)

# Split chronologically by unique timestamps.
times = np.sort(df["start_seconds"].unique())
if len(times) < 10:
    raise ValueError("Not enough distinct timestamps for a time-based split.")

cutoff_index = int(len(times) * 0.80)
cutoff_time = times[cutoff_index]

train = df[df["start_seconds"] < cutoff_time].copy()
test = df[df["start_seconds"] >= cutoff_time].copy()

if train.empty or test.empty:
    raise ValueError("Training or testing split is empty.")

print(f"Total rows: {len(df):,}")
print(f"Training rows before sampling: {len(train):,}")
print(f"Testing rows: {len(test):,}")
print(f"Training timestamps: before {cutoff_time:.2f} seconds")
print(f"Testing timestamps: from {cutoff_time:.2f} seconds")

# Limit training size to reduce memory and training time.
MAX_TRAIN_ROWS = 250_000
if len(train) > MAX_TRAIN_ROWS:
    train = train.sample(
        n=MAX_TRAIN_ROWS,
        random_state=42
    )

print(f"Training rows used: {len(train):,}")

X_train = train[FEATURES]
X_test = test[FEATURES]

all_metrics = []
predictions = pd.DataFrame({
    "start_seconds": test["start_seconds"].to_numpy(),
    "actual_cpu": test["cpu_usage"].to_numpy(),
    "predicted_cpu_baseline": test["previous_cpu"].to_numpy(),
    "actual_memory": test["memory_usage"].to_numpy(),
    "predicted_memory_baseline": test["previous_memory"].to_numpy(),
})

for name, target in TARGETS.items():
    print(f"\nTraining {name.upper()} model...")
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

    baseline_column = (
        "previous_cpu" if name == "cpu" else "previous_memory"
    )
    baseline = test[baseline_column].to_numpy()

    for method, values in [
        ("RandomForest", predicted),
        ("PersistenceBaseline", baseline),
    ]:
        all_metrics.append({
            "target": name,
            "model": method,
            "test_rows": len(y_test),
            "MAE": mean_absolute_error(y_test, values),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, values))),
            "R2": r2_score(y_test, values),
        })

    predictions[f"predicted_{name}"] = predicted

    model_path = MODEL_DIR / f"{name}_model.pkl"
    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")
    print(f"Completed in {time.time() - started:.1f} seconds")

metrics = pd.DataFrame(all_metrics)
metrics_path = OUT / "model_evaluation.csv"
predictions_path = OUT / "predictions.csv"

metrics.to_csv(metrics_path, index=False)
predictions.to_csv(predictions_path, index=False)

with open(OUT / "run_info.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": str(DATA),
        "total_rows": len(df),
        "training_rows_used": len(train),
        "testing_rows": len(test),
        "cutoff_time_seconds": float(cutoff_time),
        "features": FEATURES,
        "targets": TARGETS,
        "training_sample_limit": MAX_TRAIN_ROWS,
    }, f, indent=2)

print("\nMODEL EVALUATION")
print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nSaved evaluation:", metrics_path)
print("Saved predictions:", predictions_path)
print("Saved run information:", OUT / "run_info.json")
print("\nTraining complete.")
