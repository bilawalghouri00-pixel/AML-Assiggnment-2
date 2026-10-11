"""Task 1 - load, clean, split, standardise."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from load_data import load_raw
from tables import save_table, RESULTS

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
LABEL_RAW = "Output (S)"


def clean(df):
    """Drop useless columns, rename the label, check numbers and missing values."""
    info = {"raw_rows": df.shape[0], "raw_cols": df.shape[1]}
    if LABEL_RAW not in df.columns:
        raise KeyError(f"Label column '{LABEL_RAW}' not found. Columns: {df.columns.tolist()}")
    # a column with no information = completely empty, or the same value in every row
    removed = [c for c in df.columns
               if c != LABEL_RAW and (df[c].isna().all() or df[c].nunique(dropna=True) <= 1)]
    df = df.drop(columns=removed).rename(columns={LABEL_RAW: "fault"})
    df = df[["fault"] + FEATURES]
    for c in df.columns:
        assert pd.api.types.is_numeric_dtype(df[c]), f"{c} is not numeric"
    info["removed"] = removed
    info["missing_per_column"] = df.isna().sum().to_dict()
    df = df.dropna().reset_index(drop=True)      # (normally nothing is dropped)
    info["missing_after"] = int(df.isna().sum().sum())
    df["fault"] = df["fault"].astype(int)
    return df, info


def split_indices(n):
    """EXACT split code from the assignment."""
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def standardise(train, *others):
    mean, std = train.mean(axis=0), train.std(axis=0)   # TRAIN statistics only
    return mean, std, [(a - mean) / std for a in (train, *others)]


def prepare_all():
    raw = load_raw(str(ROOT / "data"))
    df, info = clean(raw)
    tr, va, te = split_indices(len(df))
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["fault"].to_numpy(dtype=int)
    X_sets = {"R": X, "Q": np.hstack([X, X ** 2])}      # squares BEFORE standardising
    out = {"R": {}, "Q": {}, "y": {"train": y[tr], "val": y[va], "test": y[te]},
           "stats": {}, "df": df, "info": info, "idx": (tr, va, te), "raw_X": X}
    for name, M in X_sets.items():
        mean, std, (a, b, c) = standardise(M[tr], M[va], M[te])
        out[name] = {"train": a, "val": b, "test": c}
        out["stats"][name] = (mean, std)
    return out


if __name__ == "__main__":
    raw = load_raw(str(ROOT / "data"))
    print("1.1  shape:", raw.shape)
    print("     columns:", raw.columns.tolist())
    print(raw.head(5).to_string(), "\n")

    d = prepare_all()
    df, info = d["df"], d["info"]
    print("1.2  removed columns:", info["removed"])
    print("     missing values per column (before dropping):", info["missing_per_column"])
    print("     all six features numeric: OK")
    n1, n0 = int((df.fault == 1).sum()), int((df.fault == 0).sum())
    print(f"1.3  fault=1: {n1}   fault=0: {n0}")
    tr, va, te = d["idx"]
    y = df["fault"].to_numpy()
    fr = [float(y[i].mean()) for i in (tr, va, te)]
    print(f"1.4  train={len(tr)} val={len(va)} test={len(te)}")
    print("     fault fraction train/val/test:", [round(f, 4) for f in fr])
    print("1.5  standardised train mean (should be ~0):", d["R"]["train"].mean(0).round(6))
    print("     standardised train std  (should be ~1):", d["R"]["train"].std(0).round(6))

    save_table(pd.DataFrame({
        "Item": ["Raw rows / raw columns", "Columns removed (names)",
                 "Missing values after cleaning", "Fault rows / no-fault rows",
                 "Train / validation / test rows", "Fault fraction in train / validation / test"],
        "Value": [f"{info['raw_rows']} / {info['raw_cols']}", ", ".join(info["removed"]),
                  info["missing_after"], f"{n1} / {n0}",
                  f"{len(tr)} / {len(va)} / {len(te)}",
                  " / ".join(f"{f:.4f}" for f in fr)]}), "table1_data_preparation")

    # save the standardisation numbers (the front end needs them)
    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / "scaler.npz",
             mean_R=d["stats"]["R"][0], std_R=d["stats"]["R"][1],
             mean_Q=d["stats"]["Q"][0], std_Q=d["stats"]["Q"][1])
    # two example rows (from the TEST part) for the front-end buttons
    Xt, yt = d["raw_X"][te], y[te]
    ex = {"no_fault": dict(zip(FEATURES, map(float, Xt[np.where(yt == 0)[0][0]]))),
          "fault": dict(zip(FEATURES, map(float, Xt[np.where(yt == 1)[0][0]])))}
    (RESULTS / "examples.json").write_text(json.dumps(ex, indent=2))
    print("\nSaved Table 1, scaler.npz and examples.json in results/")
