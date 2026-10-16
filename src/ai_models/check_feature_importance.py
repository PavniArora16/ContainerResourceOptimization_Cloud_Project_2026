
import joblib

names = [
    "start_seconds",
    "gap_seconds",
    "previous_cpu",
    "previous_memory",
]

base = "results/google_cluster_real/models/"

for target in ["cpu", "memory"]:
    model = joblib.load(base + target + "_model.pkl")

    print(f"\n{target.upper()} FEATURE IMPORTANCE")

    importances = sorted(
        zip(names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    for name, value in importances:
        print(f"{name}: {value:.4f}")
