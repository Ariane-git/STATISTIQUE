"""
setup.py — A executer UNE SEULE FOIS localement avant de deployer.
Entraine 6 modeles ML + calcule les valeurs SHAP.
Genere tous les fichiers .pkl necessaires.

Usage : python setup.py
Puis  : git add *.pkl *.csv requirements.txt && git commit && git push
"""

import pandas as pd
import numpy as np
import json, os, warnings, joblib
warnings.filterwarnings('ignore')

DATA_FILE = "data_cesarienne_complet1.csv"
if not os.path.exists(DATA_FILE):
    print(f"ERREUR : '{DATA_FILE}' introuvable dans le dossier courant.")
    exit(1)

print("=" * 65)
print("  SETUP — Cesarienne EDS Cameroun 2018 — 6 modeles + SHAP")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET RECODAGE
# ══════════════════════════════════════════════════════════════════════
print("\n[1/4] Chargement et recodage...")
df = pd.read_csv(DATA_FILE, encoding="utf-8")
df["number_of_antenatal_visits_during_p_m14"] = pd.to_numeric(
    df["number_of_antenatal_visits_during_p_m14"], errors="coerce")
df["number_of_antenatal_visits_during_p_m14"].fillna(
    df["number_of_antenatal_visits_during_p_m14"].mode()[0], inplace=True)
df["cesarienne"] = df["delivery_by_caesarean_section_m17"].map({"yes": 1, "no": 0})
df["visites_pn"] = pd.to_numeric(
    df["number_of_antenatal_visits_during_p_m14"], errors="coerce")

# Recodage identique au notebook ML
df["age_ml"]  = pd.cut(df["Respondent_s_current_age_v012"],
    bins=[0, 19, 34, 99], labels=["Moins_20", "20_34", "35_plus"]).astype(str)
df["mil_ml"]  = df["Type_of_place_of_residence_v025"].map({"Urban": "Urbain", "Rural": "Rural"})
df["ins_ml"]  = df["Highest_educational_level_v106"].map({
    "No education": "Aucune", "Primary": "Primaire",
    "Secondary": "Secondaire", "Higher": "Superieur"})
df["ric_ml"]  = df["Wealth_index_combined_v190"].map({
    "Poorest": "Tres_pauvre", "Poorer": "Pauvre", "Middle": "Moyen",
    "Richer": "Riche", "Richest": "Tres_riche"})
df["par_ml"]  = np.where(df["Total_children_ever_born_v201"] == 1, "Primipare", "Multipare")
df["stat_ml"] = np.where(
    df["Current_marital_status_v501"].isin(["Married", "Living with partner"]),
    "En_union", "Pas_en_union")
df["sex_ml"]  = df["sex_of_child_b4"].map({"male": "Masculin", "female": "Feminin"})
df["cva_ml"]  = np.where(df["visites_pn"] >= 4, "4_plus", "Moins_4")
df["reg_ml"]  = df["Region_v024"].astype(str)
df = df.dropna(subset=["cesarienne"])

cols = ["age_ml", "mil_ml", "ins_ml", "ric_ml", "par_ml",
        "stat_ml", "sex_ml", "cva_ml", "reg_ml"]
sub = df[cols + ["cesarienne"]].dropna()
X = pd.get_dummies(sub[cols], drop_first=False, dtype=int)
X.columns = [
    c.replace("age_ml_", "age_groupe_").replace("mil_ml_", "milieu_")
     .replace("ins_ml_", "instruction_").replace("ric_ml_", "richesse_")
     .replace("par_ml_", "parite_").replace("stat_ml_", "statut_union_")
     .replace("sex_ml_", "sexe_enfant_").replace("cva_ml_", "cva_")
     .replace("reg_ml_", "region_")
    for c in X.columns
]
y = sub["cesarienne"].astype(int)

print(f"   Dataset   : {len(df):,} obs — {y.sum()} cesariennes ({y.mean()*100:.1f}%)")
print(f"   Variables : {X.shape[1]} features apres encodage")

joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# ══════════════════════════════════════════════════════════════════════
# 2. SPLIT TRAIN / TEST
# ══════════════════════════════════════════════════════════════════════
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X.copy(), y.copy(), test_size=0.20, random_state=42, stratify=y)
ratio = int((y_train == 0).sum() / (y_train == 1).sum())
print(f"   Train     : {len(X_train):,} | Test : {len(X_test):,} | Ratio desiquilibre : {ratio}")

# Scaling pour KNN (les autres n'en ont pas besoin)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_train)
Xte_s = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# ══════════════════════════════════════════════════════════════════════
# 3. ENTRAINEMENT DES 6 MODELES
# ══════════════════════════════════════════════════════════════════════
print("\n[2/4] Entrainement des 6 modeles ML...")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Definition des 6 modeles — hyperparametres conformes au notebook
model_specs = {
    # 1. Regression Logistique (modele de reference)
    "logistic_regression": {
        "model": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),
        "scaled": True,   # utilise les donnees normalisees
        "label": "Regression Logistique ML"
    },
    # 2. Arbre de Decision
    "decision_tree": {
        "model": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42
        ),
        "scaled": False,
        "label": "Arbre de Decision"
    },
    # 3. Random Forest
    "random_forest": {
        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=30,
            min_samples_leaf=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "scaled": False,
        "label": "Random Forest"
    },
    # 4. XGBoost
    "xgboost": {
        "model": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            eval_metric="auc",
            random_state=42,
            verbosity=0
        ),
        "scaled": False,
        "label": "XGBoost"
    },
    # 5. Gradient Boosting (scikit-learn)
    "gradient_boosting": {
        "model": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        "scaled": False,
        "label": "Gradient Boosting"
    },
    # 6. K-Nearest Neighbors
    "knn": {
        "model": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            metric="euclidean"
        ),
        "scaled": True,
        "label": "K-Nearest Neighbors"
    },
}

ml_results = {}
trained_models = {}

for name, spec in model_specs.items():
    print(f"   -> {spec['label']}...", end=" ", flush=True)
    model = spec["model"]
    Xtr   = Xtr_s if spec["scaled"] else X_train
    Xte   = Xte_s if spec["scaled"] else X_test

    if name == "xgboost":
        model.fit(Xtr, y_train, eval_set=[(Xte, y_test)], verbose=False)
    else:
        model.fit(Xtr, y_train)

    proba = model.predict_proba(Xte)[:, 1]
    pred  = model.predict(Xte)

    ml_results[name] = {
        "label":     spec["label"],
        "auc":       round(float(roc_auc_score(y_test, proba)), 3),
        "f1":        round(float(f1_score(y_test, pred)), 3),
        "recall":    round(float(recall_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred)), 3),
        "accuracy":  round(float(accuracy_score(y_test, pred)), 3),
    }
    trained_models[name] = model
    joblib.dump(model, f"model_{name}.pkl")
    print(f"AUC={ml_results[name]['auc']:.3f} | "
          f"F1={ml_results[name]['f1']:.3f} | "
          f"Rappel={ml_results[name]['recall']:.3f}")

joblib.dump(ml_results, "model_results.pkl")

# Tableau recapitulatif
best_name = max(ml_results, key=lambda k: ml_results[k]["auc"])
print(f"\n   Meilleur modele : {ml_results[best_name]['label']} "
      f"(AUC={ml_results[best_name]['auc']:.3f})")

# ══════════════════════════════════════════════════════════════════════
# 4. ANALYSE SHAP
# ══════════════════════════════════════════════════════════════════════
print("\n[3/4] Calcul des valeurs SHAP (sur Random Forest)...")

import shap

# SHAP est calcule sur la Random Forest (meilleur modele tree-based stable)
# et sur un echantillon de 500 obs max pour limiter le temps de calcul
rf_model  = trained_models["random_forest"]
n_shap    = min(500, len(X_test))
X_shap    = X_test.iloc[:n_shap].copy()

explainer  = shap.TreeExplainer(rf_model)
shap_vals  = explainer.shap_values(X_shap)  # shape: (n, p, 2) ou (2, n, p)
shap_arr   = np.array(shap_vals)
print(f"   SHAP array shape : {shap_arr.shape}")

# Extraire les valeurs SHAP pour la classe 1 (cesarienne)
# Format scikit-learn RF : (n_samples, n_features, n_classes)
if shap_arr.ndim == 3 and shap_arr.shape[2] == 2:
    shap_class1 = shap_arr[:, :, 1]          # (n, p)
elif shap_arr.ndim == 3 and shap_arr.shape[0] == 2:
    shap_class1 = shap_arr[1]                 # (n, p)
else:
    shap_class1 = shap_arr                    # (n, p) direct

# Importance SHAP = moyenne des valeurs absolues
shap_importance = np.abs(shap_class1).mean(axis=0)  # (p,)

shap_df = pd.DataFrame({
    "Variable":        X.columns.tolist(),
    "SHAP_importance": shap_importance.tolist()
}).sort_values("SHAP_importance", ascending=False).reset_index(drop=True)

# Sauvegarder : importances + valeurs brutes pour beeswarm
joblib.dump({
    "shap_values":     shap_class1.tolist(),   # (n, p) — pour beeswarm
    "feature_names":   X.columns.tolist(),
    "shap_importance": shap_df.to_dict("records"),
    "X_shap":          X_shap.values.tolist(), # valeurs des features
    "n_samples":       n_shap,
}, "shap_results.pkl")

print(f"   Top 5 variables SHAP :")
for _, r in shap_df.head(5).iterrows():
    print(f"     {r['Variable']:<35} {r['SHAP_importance']:.5f}")

# ══════════════════════════════════════════════════════════════════════
# 5. IMPORTANCE GINI (RF) — pour compatibilite avec l'app existante
# ══════════════════════════════════════════════════════════════════════
print("\n[4/4] Sauvegarde importance Gini (Random Forest)...")
fi_df = pd.DataFrame({
    "Variable":   X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)
fi_df.to_csv("feature_importances.csv", index=False)
print("   feature_importances.csv sauvegarde")

# ══════════════════════════════════════════════════════════════════════
# RESUME FINAL
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  TABLEAU COMPARATIF FINAL")
print("=" * 65)
print(f"  {'Modele':<30} {'AUC':>6}  {'F1':>6}  {'Rappel':>7}")
print(f"  {'-'*30} {'-'*6}  {'-'*6}  {'-'*7}")
for name, r in sorted(ml_results.items(), key=lambda x: x[1]["auc"], reverse=True):
    star = " <-- MEILLEUR" if name == best_name else ""
    print(f"  {r['label']:<30} {r['auc']:.3f}  {r['f1']:.3f}  {r['recall']:.3f}{star}")

print("\n  Fichiers generes :")
files = ["feature_columns.pkl","scaler.pkl","model_results.pkl",
         "shap_results.pkl","feature_importances.csv"] + \
        [f"model_{n}.pkl" for n in model_specs.keys()]
for f in files:
    status = "OK" if os.path.exists(f) else "MANQUANT"
    print(f"    [{status}] {f}")

print("\n  Lancer l application :")
print("  streamlit run strem.py\n")
