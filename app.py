import importlib, importlib.util, os, subprocess, sys
import streamlit as st

# The app's environment is read-only at runtime, so missing libraries are installed into a
# writable folder and added to the END of sys.path (the preinstalled numpy/pandas keep priority).
LIBDIR = "/tmp/pylibs"
os.makedirs(LIBDIR, exist_ok=True)
if LIBDIR not in sys.path:
    sys.path.append(LIBDIR)

def _bootstrap():
    need = [(p, m) for p, m in [("scikit-learn", "sklearn"), ("joblib", "joblib"), ("matplotlib", "matplotlib")]
            if importlib.util.find_spec(m) is None]
    if not need:
        return
    pkgs = [p for p, _ in need]
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--target", LIBDIR, *pkgs],
                       capture_output=True, text=True)
    importlib.invalidate_caches()
    still = [m for _, m in need if importlib.util.find_spec(m) is None]
    if r.returncode != 0 or still:
        st.error("Could not install: " + ", ".join(pkgs) + ". Details below - please send this to your helper.")
        st.code((r.stdout + r.stderr)[-2500:]); st.stop()

_bootstrap()

import json, joblib, numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from churn_pipeline import engineer

st.set_page_config(page_title="Bank Churn Risk", layout="wide")
import os
from sklearn.ensemble import GradientBoostingClassifier

@st.cache_resource
def load_artifact():
    """Load the saved model; if the file is missing/empty/incompatible, retrain from European_Bank.csv."""
    try:
        if os.path.getsize("churn_model.joblib") > 0:
            return joblib.load("churn_model.joblib")
    except Exception:
        pass
    df = pd.read_csv("European_Bank.csv").drop(columns=["Year", "CustomerId", "Surname"], errors="ignore")
    X = engineer(df.drop(columns="Exited")); y = df["Exited"]
    m = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42).fit(X, y)
    return dict(model=m, columns=list(X.columns), threshold=0.30, name="Gradient Boosting")

art = load_artifact(); model, cols, thr = art["model"], art["columns"], art["threshold"]

@st.cache_resource
def eval_models():
    """Train on an 80/20 stratified split of European_Bank.csv and keep test-set probabilities (used when image files are missing)."""
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv("European_Bank.csv").drop(columns=["Year", "CustomerId", "Surname"], errors="ignore")
    X = engineer(df.drop(columns="Exited")); y = df["Exited"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42).fit(Xtr, ytr)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xtr, ytr),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, class_weight="balanced", random_state=42).fit(Xtr, ytr),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42).fit(Xtr, ytr),
        "Gradient Boosting": gb,
    }
    probs = {n: m.predict_proba(Xte)[:, 1] for n, m in models.items()}
    return dict(probs=probs, yte=yte.values, Xte=Xte, gb=gb)

def show_img(path, **kw):
    if os.path.exists(path):
        st.image(path, **kw); return
    if not os.path.exists("European_Bank.csv"):
        st.warning(f"Cannot draw {path}: European_Bank.csv was not found next to app.py. "
                   f"Files the app can see in {os.getcwd()}: {sorted(os.listdir('.'))}"); return
    try:
        _draw_missing(path)
    except Exception as e:
        st.error(f"Chart for {path} failed: {type(e).__name__}: {e}")

def _draw_missing(path):
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.inspection import PartialDependenceDisplay
    ev = eval_models(); name = os.path.basename(path)
    if name == "prob_dist.png":
        p, y = ev["probs"]["Gradient Boosting"], ev["yte"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(p[y == 0], bins=30, alpha=.6, label="Retained"); ax.hist(p[y == 1], bins=30, alpha=.6, label="Churned")
        ax.axvline(thr, color="k", ls="--", label=f"threshold {thr:.2f}"); ax.set_xlabel("Predicted churn probability"); ax.legend()
        ax.set_title("Probability distribution (test set)"); st.pyplot(fig)
    elif name == "roc.png":
        fig, ax = plt.subplots(figsize=(6, 5))
        for n, p in ev["probs"].items():
            f, t, _ = roc_curve(ev["yte"], p); ax.plot(f, t, label=f"{n} ({roc_auc_score(ev['yte'], p):.3f})")
        ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves (test set)"); ax.legend(fontsize=8); st.pyplot(fig)
    elif name == "pdp.png":
        fig, ax = plt.subplots(figsize=(10, 3.2))
        PartialDependenceDisplay.from_estimator(ev["gb"], ev["Xte"].astype(float), ["Age", "NumOfProducts", "Balance", "IsActiveMember"], ax=ax)
        plt.tight_layout(); st.pyplot(fig)
    else:
        st.info(f"Image not available: {path}")

def show_csv(path):
    if os.path.exists(path):
        st.dataframe(pd.read_csv(path), hide_index=True)
    else:
        st.info(f"Data file not available: {path}")

def score(row: dict) -> float:
    X = engineer(pd.DataFrame([row]))
    return float(model.predict_proba(X.reindex(columns=cols, fill_value=0))[0, 1])

def band(p):
    return "Low" if p < .2 else "Medium" if p < .5 else "High" if p < .8 else "Critical"

def inputs(prefix, d=None):
    d = d or {}
    c1, c2 = st.columns(2)
    r = {}
    r["Geography"] = c1.selectbox("Geography", ["France","Spain","Germany"], index=["France","Spain","Germany"].index(d.get("Geography","France")), key=prefix+"g")
    r["Gender"] = c2.selectbox("Gender", ["Female","Male"], index=["Female","Male"].index(d.get("Gender","Female")), key=prefix+"s")
    r["CreditScore"] = c1.slider("Credit score", 300, 850, d.get("CreditScore",650), key=prefix+"c")
    r["Age"] = c2.slider("Age", 18, 92, d.get("Age",40), key=prefix+"a")
    r["Tenure"] = c1.slider("Tenure (years)", 0, 10, d.get("Tenure",5), key=prefix+"t")
    r["Balance"] = c2.number_input("Balance", 0.0, 300000.0, float(d.get("Balance",75000.0)), 1000.0, key=prefix+"b")
    r["NumOfProducts"] = c1.slider("Number of products", 1, 4, d.get("NumOfProducts",2), key=prefix+"n")
    r["EstimatedSalary"] = c2.number_input("Estimated salary", 0.0, 250000.0, float(d.get("EstimatedSalary",100000.0)), 1000.0, key=prefix+"e")
    r["HasCrCard"] = int(c1.checkbox("Has credit card", bool(d.get("HasCrCard",1)), key=prefix+"h"))
    r["IsActiveMember"] = int(c2.checkbox("Active member", bool(d.get("IsActiveMember",1)), key=prefix+"i"))
    return r

st.title("Bank Customer Churn: Risk Scoring")
st.caption(f"Model: {art['name']} · decision threshold {thr:.2f}")
st.caption("App version: v6 (charts drawn in-app)")
t1, t2, t3, t4 = st.tabs(["Risk calculator", "What-if simulator", "Feature importance", "Probability distribution"])

with t1:
    r = inputs("calc_"); p = score(r)
    a, b, c = st.columns(3)
    a.metric("Churn probability", f"{p:.1%}"); b.metric("Risk band", band(p)); c.metric("Churn flag", "Likely to churn" if p >= thr else "Likely to stay")
    st.progress(min(p, 1.0))

with t2:
    st.write("Set a baseline customer, then change engagement and product values to see the effect.")
    l, rcol = st.columns(2)
    with l:
        st.subheader("Baseline"); base = inputs("base_")
    with rcol:
        st.subheader("Scenario")
        sc = dict(base)
        sc["NumOfProducts"] = st.slider("Number of products ", 1, 4, base["NumOfProducts"], key="sc_n")
        sc["IsActiveMember"] = int(st.checkbox("Active member ", bool(base["IsActiveMember"]), key="sc_i"))
        sc["Balance"] = st.number_input("Balance ", 0.0, 300000.0, float(base["Balance"]), 1000.0, key="sc_b")
        sc["Tenure"] = st.slider("Tenure ", 0, 10, base["Tenure"], key="sc_t")
    p0, p1 = score(base), score(sc)
    x, y, z = st.columns(3)
    x.metric("Baseline", f"{p0:.1%}"); y.metric("Scenario", f"{p1:.1%}", f"{(p1-p0)*100:+.1f} pts", delta_color="inverse"); z.metric("Scenario band", band(p1))
    ns = list(range(1, 5)); fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.plot(ns, [score({**sc, "NumOfProducts": n}) for n in ns], marker="o"); ax.set_xticks(ns)
    ax.set_xlabel("Number of products"); ax.set_ylabel("Churn probability"); ax.set_title("Sensitivity to product count (scenario customer)")
    st.pyplot(fig)

with t3:
    if os.path.exists("feature_importance.csv"):
        imp = pd.read_csv("feature_importance.csv", index_col=0).iloc[:, 0].head(12)[::-1]
        fig, ax = plt.subplots(figsize=(6, 4)); imp.plot.barh(ax=ax, color="#1f4e79"); ax.set_xlabel("Drop in ROC-AUC when shuffled")
        st.pyplot(fig)
    else:
        st.info("Data file not available: feature_importance.csv")
    show_img("figures/pdp.png", caption="Partial dependence of churn probability")
    if os.path.exists("results.json"):
        res = json.load(open("results.json"))["ablation"]
        st.write("**Behavioural vs demographic signal (ROC-AUC)**", {k: round(v, 3) for k, v in res.items()})

with t4:
    show_csv("model_comparison.csv")
    show_img("figures/prob_dist.png"); show_img("figures/roc.png", width=450)
    show_csv("risk_bands.csv")
