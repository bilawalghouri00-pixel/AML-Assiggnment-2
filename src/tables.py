"""Tiny helper: save a table as CSV and as Markdown inside results/."""
from pathlib import Path
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"

def save_table(df, name):
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    (RESULTS / f"{name}.md").write_text("\n".join(lines) + "\n")
