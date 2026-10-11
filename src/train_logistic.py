"""Task 3 - choose alpha, train final logistic models for Set R and Set Q."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prepare import prepare_all
from logistic import add_bias, cost, fit_logistic, predict
from metrics import accuracy
from tables import save_table, RESULTS

RESULTS.mkdir(exist_ok=True)
d = prepare_all()
ALPHAS = [0.01, 0.1, 1, 10]
ITERS = 2000
NAMES = ["theta0 (intercept)", "Ia", "Ib", "Ic", "Va", "Vb", "Vc",
         "Ia^2", "Ib^2", "Ic^2", "Va^2", "Vb^2", "Vc^2"]

rows, saved = [], {}
for s in ["R", "Q"]:
    Xtr, Xva = add_bias(d[s]["train"]), add_bias(d[s]["val"])
    ytr, yva = d["y"]["train"], d["y"]["val"]
    J0 = cost(Xtr, ytr, np.zeros(Xtr.shape[1]))
    print(f"\n=== Set {s} ===  J at theta=0: {J0:.4f} (should be about 0.6931)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ok, results = [], {}
    for a in ALPHAS:
        theta, hist = fit_logistic(Xtr, ytr, a, ITERS)
        always_down = bool(np.all(np.diff(np.r_[J0, hist]) <= 0))
        print(f"alpha={a:<5} final J={hist[-1]:.5f}  decreases every iteration: {always_down}")
        ax.plot(range(1, ITERS + 1), hist, label=f"alpha={a}")
        results[a] = (theta, hist)
        if always_down:
            ok.append(a)
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("J(theta)")
    ax.set_title(f"Cost vs iteration, Set {s}"); ax.legend(); fig.tight_layout()
    fig.savefig(RESULTS / f"fig2_alpha_set{s}.png", dpi=150); plt.close(fig)

    alpha = max(ok) if ok else min(ALPHAS)          # largest alpha that always goes down
    theta, hist = results[alpha]
    tr_acc = accuracy(ytr, predict(Xtr, theta))
    va_acc = accuracy(yva, predict(Xva, theta))
    print(f"CHOSEN alpha={alpha}  train acc={tr_acc:.4f}  validation acc={va_acc:.4f}")
    print("theta =", np.round(theta, 4))
    rows.append({"Feature set": f"Set {s}", "Chosen alpha": alpha,
                 "Final J(theta)": round(float(hist[-1]), 5),
                 "Train accuracy": round(tr_acc, 4), "Validation accuracy": round(va_acc, 4)})
    saved[s] = theta

save_table(pd.DataFrame(rows), "table2_logistic")
save_table(pd.DataFrame({"Weight": NAMES, "Value": np.round(saved["Q"], 5)}), "table5_theta_setQ")
np.savez(RESULTS / "logistic_models.npz", theta_R=saved["R"], theta_Q=saved["Q"])
print("\nSaved Table 2, Table 5, fig2 and logistic_models.npz in results/")
