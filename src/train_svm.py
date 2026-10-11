"""Task 4 - linear and RBF SVMs (Set R only)."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC                      # the ONLY sklearn import allowed
from prepare import prepare_all
from metrics import accuracy
from tables import save_table, RESULTS

RESULTS.mkdir(exist_ok=True)
d = prepare_all()
Xtr, Xva = d["R"]["train"], d["R"]["val"]
ytr, yva = d["y"]["train"], d["y"]["val"]
CS = [0.1, 1, 10, 100]
GAMMAS = [0.1, 1]

rows, models = [], {}
for C in CS:                                      # 4.1 linear
    m = SVC(kernel="linear", C=C).fit(Xtr, ytr)
    acc = accuracy(yva, m.predict(Xva))
    rows.append(("linear", C, "-", acc, int(m.n_support_.sum())))
    models[("linear", C, None)] = (m, acc)
    print(f"linear C={C}: val acc={acc:.4f}  support vectors={m.n_support_.sum()}")
for C in CS:                                      # 4.2 RBF
    for g in GAMMAS:
        m = SVC(kernel="rbf", C=C, gamma=g).fit(Xtr, ytr)
        acc = accuracy(yva, m.predict(Xva))
        rows.append(("rbf", C, g, acc, int(m.n_support_.sum())))
        models[("rbf", C, g)] = (m, acc)
        print(f"rbf    C={C} gamma={g}: val acc={acc:.4f}  support vectors={m.n_support_.sum()}")

tab = pd.DataFrame(rows, columns=["Kernel", "C", "gamma", "Validation accuracy", "Support vectors"])
tab["Validation accuracy"] = tab["Validation accuracy"].round(4)
save_table(tab, "table3_svm_validation")          # 4.3

best = {}                                         # 4.4 best of each kernel, saved with joblib
for kern in ["linear", "rbf"]:
    key = max((k for k in models if k[0] == kern), key=lambda k: (models[k][1], -models[k][0].n_support_.sum()))  # best accuracy; tie -> fewer support vectors
    m, acc = models[key]
    joblib.dump(m, RESULTS / f"svm_{kern}.joblib")
    best[kern] = {"C": key[1], "gamma": key[2], "val_acc": acc,
                  "support_vectors": int(m.n_support_.sum())}
    print("BEST", kern, best[kern])
(RESULTS / "svm_best.json").write_text(json.dumps(best, indent=2))

# Figure 3: validation accuracy vs C
fig, ax = plt.subplots(figsize=(7, 4.5))
lin = tab[tab.Kernel == "linear"]
ax.plot(lin["C"], lin["Validation accuracy"], "o-", label="linear")
for g in GAMMAS:
    r = tab[(tab.Kernel == "rbf") & (tab.gamma == g)]
    ax.plot(r["C"], r["Validation accuracy"], "o-", label=f"RBF gamma={g}")
ax.set_xscale("log"); ax.set_xlabel("C (log scale)"); ax.set_ylabel("validation accuracy")
ax.set_title("SVM validation accuracy vs C"); ax.legend(); fig.tight_layout()
fig.savefig(RESULTS / "fig3_svm_accuracy_vs_C.png", dpi=150); plt.close(fig)
print("Saved Table 3, fig3, svm_linear.joblib, svm_rbf.joblib in results/")
