
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# Paths and output directory
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

predictions_path = RESULTS_DIR / "predictions.csv"

# --------------------------------------------------
# Load and validate predictions
# --------------------------------------------------

if not predictions_path.exists():
    raise FileNotFoundError(
        f"Predictions file not found: {predictions_path}\n"
        "Generate predictions before running evaluation."
    )

df = pd.read_csv(predictions_path)

required_columns = [
    "next_cpu",
    "predicted_cpu",
    "next_memory",
    "predicted_memory"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in predictions.csv: {missing_columns}"
    )

if df.empty:
    raise ValueError("predictions.csv contains no rows.")

# Remove rows where actual or predicted values are missing.
df = df.dropna(subset=required_columns).copy()

if len(df) < 2:
    raise ValueError(
        "At least two valid observations are required for evaluation."
    )

# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_model(y_actual, y_predicted, model_name):
    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    r2 = r2_score(y_actual, y_predicted)

    print(f"\n{'=' * 45}")
    print(f"{model_name} Evaluation")
    print(f"{'=' * 45}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2:   {r2:.6f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Evaluated Observations": len(y_actual)
    }

# --------------------------------------------------
# Evaluate CPU and memory prediction models
# --------------------------------------------------

cpu_results = evaluate_model(
    df["next_cpu"],
    df["predicted_cpu"],
    "CPU Random Forest"
)

memory_results = evaluate_model(
    df["next_memory"],
    df["predicted_memory"],
    "Memory Random Forest"
)

metrics = pd.DataFrame([cpu_results, memory_results])

print("\nMODEL COMPARISON")
print(metrics.to_string(index=False))

metrics_path = RESULTS_DIR / "model_evaluation.csv"
metrics.to_csv(metrics_path, index=False)

print(f"\nMetrics saved to: {metrics_path}")

# --------------------------------------------------
# Actual vs predicted line graphs
# Retains the original graph filenames.
# --------------------------------------------------

def plot_line_predictions(
    actual,
    predicted,
    resource_name,
    filename
):
    actual_values = np.asarray(actual)
    predicted_values = np.asarray(predicted)

    # Display up to 200 observations for readability.
    count = min(200, len(actual_values))

    plt.figure(figsize=(10, 5))
    plt.plot(
        actual_values[:count],
        label=f"Actual {resource_name}",
        linewidth=2
    )
    plt.plot(
        predicted_values[:count],
        label=f"Predicted {resource_name}",
        linewidth=2
    )

    plt.xlabel("Test observation")
    plt.ylabel(f"{resource_name} usage")
    plt.title(f"Actual vs Predicted {resource_name} Usage")
    plt.legend()
    plt.tight_layout()

    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

plot_line_predictions(
    df["next_cpu"],
    df["predicted_cpu"],
    "CPU",
    "cpu_prediction.png"
)

plot_line_predictions(
    df["next_memory"],
    df["predicted_memory"],
    "Memory",
    "memory_prediction.png"
)

# --------------------------------------------------
# Actual vs predicted scatter plots
# The dashed diagonal represents perfect predictions.
# --------------------------------------------------

def plot_scatter_predictions(
    actual,
    predicted,
    model_name,
    filename
):
    actual_values = np.asarray(actual)
    predicted_values = np.asarray(predicted)

    lower = min(actual_values.min(), predicted_values.min())
    upper = max(actual_values.max(), predicted_values.max())

    plt.figure(figsize=(7, 5))
    plt.scatter(
        actual_values,
        predicted_values,
        alpha=0.5
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()

    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

plot_scatter_predictions(
    df["next_cpu"],
    df["predicted_cpu"],
    "CPU Random Forest",
    "cpu_actual_vs_predicted.png"
)

plot_scatter_predictions(
    df["next_memory"],
    df["predicted_memory"],
    "Memory Random Forest",
    "memory_actual_vs_predicted.png"
)

# --------------------------------------------------
# Completion summary
# --------------------------------------------------

print("\nEvaluation graphs generated.")
print(f"Valid observations evaluated: {len(df)}")
print(f"Results directory: {RESULTS_DIR}")
print("Evaluation complete.")
