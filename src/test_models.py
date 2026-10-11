"""Task 5 - evaluate the four final models on the TEST rows (once)."""
import json
import joblib
import numpy as np
import pandas as pd
from prepare import prepare_all
from logistic import add_bias, predict
from metrics import accuracy, confusion_matrix
from tables import save_table, RESULTS

d = prepare_all()
y = d["y"]["test"]
lr = np.load(RESULTS / "logistic_models.npz")
best = json.loads((RESULTS / "svm_best.json").read_text())
svm_lin = joblib.load(RESULTS / "svm_linear.joblib")
svm_rbf = joblib.load(RESULTS / "svm_rbf.joblib")

preds = {
    "Logistic regression, Set R": predict(add_bias(d["R"]["test"]), lr["theta_R"]),
    "Logistic regression, Set Q": predict(add_bias(d["Q"]["test"]), lr["theta_Q"]),
    f"Linear SVM (C={best['linear']['C']})": svm_lin.predict(d["R"]["test"]),
    f"RBF SVM (C={best['rbf']['C']}, gamma={best['rbf']['gamma']})": svm_rbf.predict(d["R"]["test"]),
}
rows = []
for name, p in preds.items():
    cm = confusion_matrix(y, p)
    assert cm.sum() == len(y), "confusion matrix must add up to the number of test rows"
    print(f"\n{name}\n accuracy={accuracy(y, p):.4f}\n confusion matrix [[TN, FP],[FN, TP]]:\n{cm}")
    rows.append({"Model": name, "Accuracy": round(accuracy(y, p), 4),
                 "TN": cm[0, 0], "False alarms": cm[0, 1],
                 "Missed faults": cm[1, 0], "TP": cm[1, 1]})
save_table(pd.DataFrame(rows), "table4_test")
print("\nTest rows:", len(y), "- Saved Table 4 in results/")
