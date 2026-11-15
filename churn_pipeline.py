"""Bank churn: preprocessing, feature engineering, modelling, evaluation, explainability."""
import json, warnings, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
warnings.filterwarnings("ignore")
SEED = 42
DATA = "/mnt/user-data/uploads/European_Bank.csv"

def engineer(df):
    """Shared by training and the Streamlit app so both use identical features."""
    d = df.copy()
    d["BalanceSalaryRatio"] = d["Balance"] / (d["EstimatedSalary"] + 1)   # +1 guards divide-by-zero
    d["ProductDensity"] = d["NumOfProducts"] / (d["Tenure"] + 1)           # products per year of relationship
    d["EngagementProduct"] = d["IsActiveMember"] * d["NumOfProducts"]
    d["AgeTenure"] = d["Age"] * d["Tenure"]
    d["ZeroBalance"] = (d["Balance"] == 0).astype(int)
    d = pd.get_dummies(d, columns=["Geography", "Gender"], dtype=int)
    for c in ["Geography_France", "Geography_Germany", "Geography_Spain", "Gender_Female", "Gender_Male"]:
        if c not in d: d[c] = 0
    return d

if __name__ == "__main__":
    raw = pd.read_csv(DATA)
    df = raw.drop(columns=["Year", "CustomerId", "Surname"])   # Year is constant (2025); IDs are non-informative
    assert df.isna().sum().sum() == 0
    X = engineer(df.drop(columns="Exited")); y = df["Exited"]
    cols = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print("churn rate train/test:", ytr.mean().round(4), yte.mean().round(4))

    num = ["CreditScore","Age","Tenure","Balance","EstimatedSalary","BalanceSalaryRatio","ProductDensity","AgeTenure"]
    scale = ColumnTransformer([("s", StandardScaler(), num)], remainder="passthrough")
    models = {
      "Logistic Regression": Pipeline([("p", scale), ("m", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))]),
      "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, class_weight="balanced", random_state=SEED),
      "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
      "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=(ytr==0).sum()/(ytr==1).sum(), eval_metric="logloss", random_state=SEED)
    except ImportError:
        print("xgboost not installed; skipped (optional per brief)")

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted, probs = [], {}, {}
    for n, m in models.items():
        cv = cross_val_score(m, Xtr, ytr, cv=skf, scoring="roc_auc").mean()
        m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; pred = (p >= .5).astype(int)
        fitted[n], probs[n] = m, p
        rows.append(dict(Model=n, Accuracy=accuracy_score(yte,pred), Precision=precision_score(yte,pred),
                         Recall=recall_score(yte,pred), F1=f1_score(yte,pred), ROC_AUC=roc_auc_score(yte,p), CV_AUC_5fold=cv))
    res = pd.DataFrame(rows).round(4); print(res.to_string(index=False))
    res.to_csv("model_comparison.csv", index=False)

    # choose best by CV AUC (not test) to avoid test-set selection
    best = res.sort_values("CV_AUC_5fold", ascending=False).iloc[0]["Model"]; print("Selected:", best)
    bm, bp = fitted[best], probs[best]

    # threshold tuning: best F1 on out-of-fold train predictions, evaluated on test
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(bm, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    ths = np.linspace(.1, .8, 71); f1s = [f1_score(ytr, oof >= t) for t in ths]
    thr = float(ths[int(np.argmax(f1s))]); pred = (bp >= thr).astype(int)
    tuned = dict(threshold=round(thr,2), accuracy=accuracy_score(yte,pred), precision=precision_score(yte,pred),
                 recall=recall_score(yte,pred), f1=f1_score(yte,pred), roc_auc=roc_auc_score(yte,bp),
                 confusion=confusion_matrix(yte,pred).tolist())
    print("Tuned:", {k: (round(v,4) if isinstance(v,float) else v) for k,v in tuned.items()})

    # risk bands
    bands = pd.cut(bp, [0,.2,.5,.8,1.0001], labels=["Low","Medium","High","Critical"], include_lowest=True)
    band_tbl = pd.DataFrame({"band": bands, "y": yte.values}).groupby("band", observed=True)["y"].agg(customers="count", actual_churn_rate="mean").round(3)
    print(band_tbl); band_tbl.to_csv("risk_bands.csv")

    # explainability: permutation importance (+ SHAP if available)
    pi = permutation_importance(bm, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False); imp.round(4).to_csv("feature_importance.csv")
    print(imp.head(10).round(4))
    try:
        import shap
        ex = shap.Explainer(bm.predict_proba if not hasattr(bm,"feature_importances_") else bm, Xtr.sample(500, random_state=SEED))
        sv = ex(Xte.sample(500, random_state=SEED)); shap.summary_plot(sv, show=False); plt.tight_layout()
        plt.savefig("figures/shap_summary.png", dpi=150); plt.close()
    except ImportError:
        print("shap not installed; permutation importance + PDP used instead")

    # figures
    plt.figure(figsize=(6,5))
    for n,p in probs.items():
        f,t,_ = roc_curve(yte,p); plt.plot(f,t,label=f"{n} ({roc_auc_score(yte,p):.3f})")
    plt.plot([0,1],[0,1],"k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figures/roc.png", dpi=150); plt.close()
    plt.figure(figsize=(6,5)); imp.head(12)[::-1].plot.barh(color="#1f4e79"); plt.title(f"Permutation importance (ΔAUC) – {best}")
    plt.tight_layout(); plt.savefig("figures/importance.png", dpi=150); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(bp[yte==0],bins=30,alpha=.6,label="Retained"); plt.hist(bp[yte==1],bins=30,alpha=.6,label="Churned")
    plt.axvline(thr,color="k",ls="--",label=f"threshold {thr:.2f}"); plt.xlabel("Predicted churn probability"); plt.legend(); plt.title("Probability distribution (test set)")
    plt.tight_layout(); plt.savefig("figures/prob_dist.png", dpi=150); plt.close()
    top = [c for c in ["Age","NumOfProducts","Balance","IsActiveMember"] if c in cols]
    fig, ax = plt.subplots(figsize=(10,3.2)); PartialDependenceDisplay.from_estimator(bm, Xte.astype(float), top, ax=ax); plt.tight_layout()
    plt.savefig("figures/pdp.png", dpi=150); plt.close()

    # behavioural vs demographic signal (for the report's framing)
    demo = ["Age","Gender_Female","Gender_Male","Geography_France","Geography_Germany","Geography_Spain"]
    beh = [c for c in cols if c not in demo and c != "CreditScore"]
    def auc_for(fs):
        m = GradientBoostingClassifier(n_estimators=200, learning_rate=.05, max_depth=3, subsample=.8, random_state=SEED).fit(Xtr[fs], ytr)
        return roc_auc_score(yte, m.predict_proba(Xte[fs])[:,1])
    ablation = {"all_features": auc_for(cols), "behavioural_only": auc_for(beh), "demographic_only": auc_for(demo)}
    print("Ablation AUC:", {k: round(v,4) for k,v in ablation.items()})

    # EDA figure
    fig, ax = plt.subplots(1,3,figsize=(12,3.4))
    for a,(c,t) in zip(ax,[("NumOfProducts","Churn rate by products"),("Geography","by country"),("IsActiveMember","by activity")]):
        df.groupby(c)["Exited"].mean().plot.bar(ax=a,color="#1f4e79"); a.set_title(t); a.set_ylabel("churn rate")
    plt.tight_layout(); plt.savefig("figures/eda.png", dpi=150); plt.close()

    joblib.dump(dict(model=bm, columns=cols, threshold=thr, name=best), "churn_model.joblib")
    json.dump(dict(selected=best, tuned=tuned, ablation=ablation), open("results.json","w"), indent=2, default=float)
