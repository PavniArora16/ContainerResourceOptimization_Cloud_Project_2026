
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "results" / "google_cluster_real_v2" / "models"
OUTPUT_DIR = ROOT / "results" / "google_cluster_real_v2"

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

TARGETS = ["cpu_usage", "memory_usage"]


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.ai_models.evaluate_user_file_v2 "
            '"path\\to\\input.csv"'
        )
        return

    input_path = Path(sys.argv[1]).expanduser()

    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print("ERROR: Input CSV is missing required feature columns:")
        print(", ".join(missing))
        print("\nRequired columns:")
        print(", ".join(FEATURES))
        return

    df = df.replace([np.inf, -np.inf], np.nan)
    valid = df.dropna(subset=FEATURES).copy()

    if valid.empty:
        print("ERROR: No rows have all required feature values.")
        return

    cpu_model = joblib.load(MODEL_DIR / "cpu_model_v2.pkl")
    memory_model = joblib.load(MODEL_DIR / "memory_model_v2.pkl")

    X = valid[FEATURES]

    valid["predicted_cpu"] = np.maximum(cpu_model.predict(X), 0)
    valid["predicted_memory"] = np.maximum(
        memory_model.predict(X), 0
    )

    print("\n========== USER FILE PREDICTIONS ==========")
    print(f"Input file: {input_path}")
    print(f"Input rows: {len(df)}")
    print(f"Rows predicted: {len(valid)}")

    # Metrics are valid only when actual target values are supplied.
    metric_rows = []

    if all(c in valid.columns for c in TARGETS):
        for target, prediction_col in [
            ("cpu_usage", "predicted_cpu"),
            ("memory_usage", "predicted_memory"),
        ]:
            evaluation = valid.dropna(subset=[target])
            evaluation = evaluation[
                np.isfinite(evaluation[target])
            ]

            if len(evaluation) < 2:
                print(
                    f"\nNot enough actual {target} values "
                    "to calculate reliable metrics."
                )
                continue

            actual = evaluation[target]
            predicted = evaluation[prediction_col]

            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)

            metric_rows.append({
                "target": target,
                "rows_evaluated": len(evaluation),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            })

        if metric_rows:
            metrics = pd.DataFrame(metric_rows)
            print("\n========== EVALUATION METRICS ==========")
            print(metrics.to_string(index=False))
        else:
            metrics = pd.DataFrame()
    else:
        metrics = pd.DataFrame()
        print("\nActual cpu_usage and/or memory_usage columns are absent.")
        print("Predictions can be generated, but evaluation metrics")
        print("cannot be calculated without the actual target values.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{input_path.stem}_predictions.csv"
    valid.to_csv(output_path, index=False)

    print(f"\nPredictions saved to: {output_path}")

    if not metrics.empty:
        metrics_path = OUTPUT_DIR / f"{input_path.stem}_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        print(f"Metrics saved to: {metrics_path}")

    print("\nFirst 10 predictions:")
    print(
        valid[
            ["predicted_cpu", "predicted_memory"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
