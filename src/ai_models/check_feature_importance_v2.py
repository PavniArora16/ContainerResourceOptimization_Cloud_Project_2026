
from pathlib import Path
import pandas as pd

base = Path("results/google_cluster_real_v2")

for target in ["cpu", "memory"]:
    path = base / f"{target}_feature_importance.csv"

    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)
    df = df.sort_values("importance", ascending=False)

    print(f"\n{target.upper()} FEATURE IMPORTANCE")
    print(df.to_string(index=False, formatters={
        "importance": lambda x: f"{x:.4f}"
    }))
