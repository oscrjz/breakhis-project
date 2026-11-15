"""Gather results/*.json into one table (results/summary.csv and results/summary.md).

Runs that share a name apart from a trailing _s<seed> are averaged, with the standard deviation.
"""
import json
import re
from pathlib import Path

import pandas as pd

COLS = ["balanced_accuracy", "auroc", "benign_recall", "macro_f1", "accuracy",
        "patient_recognition_rate", "patient_accuracy"]


def main(results_dir="results"):
    rows = []
    for f in sorted(Path(results_dir).glob("*.json")):
        r = json.loads(f.read_text())
        rows.append({"experiment": re.sub(r"_s\d+$", "", r["run_name"]), "seed": r["config"]["seed"],
                     "model": r["config"]["model"], "params_M": r["params"] / 1e6,
                     "cpu_ms": r["cpu_ms_per_image"], **r["test"]})
    if not rows:
        print("No results yet.")
        return
    df = pd.DataFrame(rows)
    agg = df.groupby(["experiment", "model"]).agg(
        runs=("seed", "count"), params_M=("params_M", "first"), cpu_ms=("cpu_ms", "mean"),
        **{f"{c}_mean": (c, "mean") for c in COLS}, **{f"{c}_std": (c, "std") for c in COLS}).reset_index()
    agg.to_csv(Path(results_dir) / "summary.csv", index=False)
    table = agg[["experiment", "model", "runs", "params_M", "cpu_ms"]].copy()
    for c in COLS:
        table[c] = [f"{m:.3f} ± {s:.3f}" if pd.notna(s) else f"{m:.3f}"
                    for m, s in zip(agg[f"{c}_mean"], agg[f"{c}_std"])]
    md = table.round({"params_M": 1, "cpu_ms": 1}).to_markdown(index=False)
    (Path(results_dir) / "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
