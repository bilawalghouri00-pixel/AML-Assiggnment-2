"""Task 7.1 - tiny web page. Run:  streamlit run app/app.py"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logistic import add_bias, hypothesis, predict      # our own code
RES = ROOT / "results"
KEYS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


@st.cache_resource
def load_models():
    lr = np.load(RES / "logistic_models.npz")
    sc = np.load(RES / "scaler.npz")
    svm = joblib.load(RES / "svm_rbf.joblib")
    return lr["theta_Q"], sc, svm


theta_Q, sc, svm = load_models()
ex_file = RES / "examples.json"
ex = json.loads(ex_file.read_text()) if ex_file.exists() else {
    "no_fault": dict(zip(KEYS, [0.0] * 6)), "fault": dict(zip(KEYS, [0.0] * 6))}
for k in KEYS:
    st.session_state.setdefault(k, ex["no_fault"][k])


def load_example(name):
    for k in KEYS:
        st.session_state[k] = ex[name][k]


st.title("Power line fault detector")
st.write("Type one instant of the 3 currents (A) and 3 voltages (per unit), "
         "then compare the two saved models.")
b1, b2 = st.columns(2)
b1.button("Fill with a no-fault example", on_click=load_example, args=("no_fault",))
b2.button("Fill with a fault example", on_click=load_example, args=("fault",))

c1, c2 = st.columns(2)
for i, k in enumerate(KEYS):
    (c1 if i < 3 else c2).number_input(k, key=k, format="%.5f")

if st.button("Predict", type="primary"):
    x = np.array([[st.session_state[k] for k in KEYS]], dtype=float)
    # Set Q logistic regression: raw + squares, standardise, add x0 = 1
    xq = (np.hstack([x, x ** 2]) - sc["mean_Q"]) / sc["std_Q"]
    p = float(hypothesis(add_bias(xq), theta_Q)[0])
    lr_pred = int(predict(add_bias(xq), theta_Q)[0])
    # RBF SVM: raw standardised (Set R)
    svm_pred = int(svm.predict((x - sc["mean_R"]) / sc["std_R"])[0])

    left, right = st.columns(2)
    with left:
        st.subheader("Logistic regression (Set Q)")
        if lr_pred == 1:
            st.error("FAULT") 
        else:
             st.success("NO FAULT")
        st.write(f"probability of fault: {p:.3f}")
    with right:
        st.subheader("RBF SVM")
        if svm_pred == 1:
            st.error("FAULT")  
        else:
             st.success("NO FAULT")
