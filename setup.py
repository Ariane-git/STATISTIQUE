"""
setup.py — À exécuter UNE SEULE FOIS avant de lancer l'application Streamlit.
Génère tous les fichiers nécessaires (modèles .pkl et fichiers .json).

Conforme exactement au notebook : machine_learning_cesarienne.ipynb
                                 + cesarienne_eds_cameroun.ipynb

Usage :
    python setup.py

Ensuite lancer l'app :
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import json
import warnings
import os
warnings.filterwarnings('ignore')

# ── Vérification du fichier de données ────────────────────────────────────────
DATA_FILE = "data_cesarienne_complet1.csv"
if not os.path.exists(DATA_FILE):
    print(f"ERREUR : Le fichier '{DATA_FILE}' est introuvable.")
    print("   Placez ce fichier dans le même dossier que setup.py et relancez.")
    exit(1)

print("=" * 60)
print("  SETUP — Prédiction Césarienne EDS Cameroun 2018")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET PRÉTRAITEMENT
# Conforme à : cesarienne_eds_cameroun.ipynb — sections 2 et 3
# ══════════════════════════════════════════════════════════════════════
print("\n[1/5] Chargement et prétraitement des données...")

df = pd.read_csv(DATA_FILE)

# Valeurs manquantes (section 3.1 du notebook EDS)
df['number_of_antenatal_visits_during_p_m14'] = pd.to_numeric(
    df['number_of_antenatal_visits_during_p_m14'], errors='coerce')
df['number_of_antenatal_visits_during_p_m14'].fillna(
    df['number_of_antenatal_visits_during_p_m14'].mode()[0], inplace=True)

df['current_age_of_child_b8'] = pd.to_numeric(
    df['current_age_of_child_b8'], errors='coerce')
df['current_age_of_child_b8'].fillna(
    df['current_age_of_child_b8'].mean(), inplace=True)

# Variable dépendante
df['cesarienne'] = df['delivery_by_caesarean_section_m17'].map({'yes': 1, 'no': 0})

# ── Recodage pour la RÉGRESSION LOGISTIQUE (notebook EDS, section 3.2) ──
# Labels lisibles avec accents
df['age_groupe'] = pd.cut(df['Respondent_s_current_age_v012'],
    bins=[0, 19, 34, 99],
    labels=['Moins de 20 ans', '20-34 ans', '35 ans et plus']).astype(str)

df['milieu'] = df['Type_of_place_of_residence_v025'].map(
    {'Urban': 'Urbain', 'Rural': 'Rural'})

df['instruction'] = df['Highest_educational_level_v106'].map({
    'No education': 'Aucune', 'Primary': 'Primaire',
    'Secondary': 'Secondaire', 'Higher': 'Supérieur'})

df['richesse'] = df['Wealth_index_combined_v190'].map({
    'Poorest': 'Très pauvre', 'Poorer': 'Pauvre', 'Middle': 'Moyen',
    'Richer': 'Riche', 'Richest': 'Très riche'})

df['parite'] = np.where(df['Total_children_ever_born_v201'] == 1, 'Primipare', 'Multipare')

df['statut_union'] = np.where(
    df['Current_marital_status_v501'].isin(['Married', 'Living with partner']),
    'En union', 'Pas en union')

df['sexe_enfant'] = df['sex_of_child_b4'].map({'male': 'Masculin', 'female': 'Féminin'})

df['visites_pn'] = pd.to_numeric(
    df['number_of_antenatal_visits_during_p_m14'], errors='coerce')
df['cva'] = np.where(df['visites_pn'] >= 4, '4 visites et plus', 'Moins de 4 visites')

df['region'] = df['Region_v024'].astype(str)

# ── Recodage SUPPLÉMENTAIRE pour l'analyse descriptive (notebook EDS) ──
df['grossesse_desiree'] = df['Wanted_last_child_v367'].map({
    'Wanted then': 'Désirée alors',
    'Wanted later': 'Désirée plus tard',
    'Wanted no more': 'Non désirée'})

def classer_lieu(lieu):
    if pd.isna(lieu): return np.nan
    lieu = lieu.lower()
    if 'home' in lieu:                                                    return 'Domicile'
    elif 'government' in lieu or 'sub-divisional' in lieu or 'public' in lieu: return 'Secteur public'
    elif 'integrated' in lieu or 'dispensary' in lieu:                   return 'Secteur public'
    elif 'confessional' in lieu:                                          return 'Confessionnel'
    elif 'private' in lieu or 'clinic' in lieu or 'cabinet' in lieu:     return 'Secteur privé'
    else:                                                                  return 'Autre'
df['lieu_accouchement'] = df['place_of_delivery_m15'].apply(classer_lieu)

df = df.dropna(subset=['cesarienne'])

print(f"   Dataset prêt : {len(df):,} observations")
print(f"   Césariennes  : {int(df['cesarienne'].sum())} ({df['cesarienne'].mean()*100:.1f}%)")
print(f"   Voie basse   : {int((df['cesarienne']==0).sum())} ({(df['cesarienne']==0).mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# 2. ANALYSE BIVARIÉE (Chi²)
# Conforme à : cesarienne_eds_cameroun.ipynb — section 5
# ══════════════════════════════════════════════════════════════════════
print("\n[2/5] Analyse bivariée (Chi²)...")
from scipy.stats import chi2_contingency

vars_biv = {
    'age_groupe':        "Groupe d'âge",
    'milieu':            'Milieu de résidence',
    'instruction':       "Niveau d'instruction",
    'richesse':          'Indice de richesse',
    'parite':            'Parité',
    'statut_union':      'Statut matrimonial',
    'grossesse_desiree': 'Grossesse désirée',
    'sexe_enfant':       "Sexe de l'enfant",
    'cva':               'Visites prénatales',
    'lieu_accouchement': "Lieu d'accouchement",
    'region':            'Région'
}

biv_results = {}
for var, label in vars_biv.items():
    sub = df[[var, 'cesarienne']].dropna()
    ct     = pd.crosstab(sub[var], sub['cesarienne'])
    ct_pct = pd.crosstab(sub[var], sub['cesarienne'], normalize='index') * 100
    chi2, p, dof, _ = chi2_contingency(ct)
    tab_data = {}
    for mod in ct.index:
        tab_data[str(mod)] = {
            'n':       int(ct.loc[mod].sum()),
            'n_ces':   int(ct.loc[mod, 1]) if 1 in ct.columns else 0,
            'pct_ces': round(float(ct_pct.loc[mod, 1]) if 1 in ct_pct.columns else 0.0, 1)
        }
    biv_results[var] = {
        'label':   label,
        'p_value': round(float(p), 4),
        'chi2':    round(float(chi2), 2),
        'data':    tab_data
    }

with open('bivariate_results.json', 'w', encoding='utf-8') as f:
    json.dump(biv_results, f, ensure_ascii=True, indent=2)
print("   bivariate_results.json généré")

# ══════════════════════════════════════════════════════════════════════
# 3. RÉGRESSION LOGISTIQUE MULTIVARIÉE
# Conforme à : cesarienne_eds_cameroun.ipynb — section 7
# Variables : age_groupe, milieu, instruction, richesse, parite,
#             statut_union, sexe_enfant, cva
# ══════════════════════════════════════════════════════════════════════
print("\n[3/5] Régression logistique multivariée...")
import statsmodels.api as sm

vars_logit = ['age_groupe', 'milieu', 'instruction', 'richesse',
              'parite', 'statut_union', 'sexe_enfant', 'cva']

df_logit   = df[vars_logit + ['cesarienne']].dropna()
X_logit    = pd.get_dummies(df_logit[vars_logit], drop_first=True).astype(float)
X_logit_c  = sm.add_constant(X_logit)
y_logit    = df_logit['cesarienne'].astype(float)

result = sm.Logit(y_logit, X_logit_c).fit(disp=0)
print(f"   Pseudo R² = {result.prsquared:.3f}  |  AIC = {result.aic:.1f}  |  N = {int(result.nobs)}")

logit_data = []
for var_name in result.params.index:
    coef    = float(result.params[var_name])
    pval    = float(result.pvalues[var_name])
    ci_low  = float(result.conf_int().loc[var_name, 0])
    ci_high = float(result.conf_int().loc[var_name, 1])
    logit_data.append({
        'variable':    str(var_name),
        'coef':        round(coef, 4),
        'OR':          round(float(np.exp(coef)), 3),
        'CI_low':      round(float(np.exp(ci_low)), 3),
        'CI_high':     round(float(np.exp(ci_high)), 3),
        'p_value':     round(pval, 4),
        'significant': bool(pval < 0.05)
    })

with open('logit_results.json', 'w', encoding='utf-8') as f:
    json.dump(logit_data, f, ensure_ascii=True, indent=2)
print(f"   logit_results.json généré ({len(logit_data)} variables)")

# ══════════════════════════════════════════════════════════════════════
# 4. PRÉPARATION FEATURES ML
# Conforme à : machine_learning_cesarienne.ipynb — section 3
# Labels sans accents (format exact du notebook ML)
# ══════════════════════════════════════════════════════════════════════
print("\n[4/5] Préparation des features ML...")

# Recodage ML — labels SANS accents (identique au notebook ML)
df['age_groupe_ml'] = pd.cut(df['Respondent_s_current_age_v012'],
    bins=[0, 19, 34, 99],
    labels=['Moins_20', '20_34', '35_plus']).astype(str)

df['milieu_ml']        = df['Type_of_place_of_residence_v025'].map({'Urban': 'Urbain', 'Rural': 'Rural'})
df['instruction_ml']   = df['Highest_educational_level_v106'].map({
    'No education': 'Aucune', 'Primary': 'Primaire',
    'Secondary': 'Secondaire', 'Higher': 'Superieur'})
df['richesse_ml']      = df['Wealth_index_combined_v190'].map({
    'Poorest': 'Tres_pauvre', 'Poorer': 'Pauvre', 'Middle': 'Moyen',
    'Richer': 'Riche', 'Richest': 'Tres_riche'})
df['parite_ml']        = np.where(df['Total_children_ever_born_v201'] == 1, 'Primipare', 'Multipare')
df['statut_union_ml']  = np.where(
    df['Current_marital_status_v501'].isin(['Married', 'Living with partner']),
    'En_union', 'Pas_en_union')
df['sexe_enfant_ml']   = df['sex_of_child_b4'].map({'male': 'Masculin', 'female': 'Feminin'})
df['cva_ml']           = np.where(df['visites_pn'] >= 4, '4_plus', 'Moins_4')
df['region_ml']        = df['Region_v024'].astype(str)

vars_ml = ['age_groupe_ml', 'milieu_ml', 'instruction_ml', 'richesse_ml',
           'parite_ml', 'statut_union_ml', 'sexe_enfant_ml', 'cva_ml', 'region_ml']

df_ml = df[vars_ml + ['cesarienne']].dropna()
X = pd.get_dummies(df_ml[vars_ml], drop_first=False, dtype=int)

# Nettoyer les noms de colonnes (retirer le suffixe _ml_)
X.columns = [c.replace('age_groupe_ml_', 'age_groupe_')
               .replace('milieu_ml_', 'milieu_')
               .replace('instruction_ml_', 'instruction_')
               .replace('richesse_ml_', 'richesse_')
               .replace('parite_ml_', 'parite_')
               .replace('statut_union_ml_', 'statut_union_')
               .replace('sexe_enfant_ml_', 'sexe_enfant_')
               .replace('cva_ml_', 'cva_')
               .replace('region_ml_', 'region_')
             for c in X.columns]

y = df_ml['cesarienne'].astype(int)

print(f"   Variables après encodage : {X.shape[1]}")
print(f"   Observations utilisées   : {X.shape[0]:,}")
print(f"   Voie basse (0) : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"   Césarienne (1) : {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

with open('feature_columns.json', 'w', encoding='utf-8') as f:
    json.dump(X.columns.tolist(), f, ensure_ascii=True)
print(f"   feature_columns.json généré")

# ══════════════════════════════════════════════════════════════════════
# 5. ENTRAÎNEMENT DES 3 MODÈLES ML
# Conforme EXACTEMENT à : machine_learning_cesarienne.ipynb
# sections 4 (Decision Tree), 5 (Random Forest), 6 (XGBoost)
# ══════════════════════════════════════════════════════════════════════
print("\n[5/5] Entraînement des 3 modèles ML...")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                              precision_score, accuracy_score,
                              classification_report, roc_curve)
import joblib

# Division 80/20 avec stratify — identique au notebook ML section 3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"   Entraînement : {X_train.shape[0]:,} femmes")
print(f"   Test         : {X_test.shape[0]:,} femmes")

# Ratio déséquilibre (calculé sur y_train comme dans le notebook)
ratio = int((y_train == 0).sum() / (y_train == 1).sum())
print(f"   Ratio déséquilibre : {ratio} (1 césar. pour {ratio} voies basses)")

# ── Decision Tree (section 4 du notebook ML) ──────────────────────────
print("\n   → Decision Tree...", end=" ", flush=True)
dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)
dt.fit(X_train, y_train)
dt_proba = dt.predict_proba(X_test)[:, 1]
dt_pred  = dt.predict(X_test)
dt_auc   = roc_auc_score(y_test, dt_proba)
dt_f1    = f1_score(y_test, dt_pred)
dt_rec   = recall_score(y_test, dt_pred)
joblib.dump(dt, 'model_decision_tree.pkl')
print(f"AUC={dt_auc:.3f} | F1={dt_f1:.3f} | Rappel={dt_rec:.3f} ✅")

# ── Random Forest (section 5 du notebook ML) ──────────────────────────
print("   → Random Forest...", end=" ", flush=True)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=30,
    min_samples_leaf=15,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred  = rf.predict(X_test)
rf_auc   = roc_auc_score(y_test, rf_proba)
rf_f1    = f1_score(y_test, rf_pred)
rf_rec   = recall_score(y_test, rf_pred)
joblib.dump(rf, 'model_random_forest.pkl')
print(f"AUC={rf_auc:.3f} | F1={rf_f1:.3f} | Rappel={rf_rec:.3f} ✅")

# ── XGBoost (section 6 du notebook ML) ───────────────────────────────
# Paramètres EXACTS du notebook : subsample=0.8, colsample_bytree=0.8,
# eval_metric='auc', eval_set=[(X_test, y_test)]
print("   → XGBoost...", end=" ", flush=True)
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,           # présent dans le notebook
    colsample_bytree=0.8,    # présent dans le notebook
    scale_pos_weight=ratio,
    eval_metric='auc',       # identique au notebook
    random_state=42,
    verbosity=0
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # identique au notebook
    verbose=False
)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_pred  = xgb.predict(X_test)
xgb_auc   = roc_auc_score(y_test, xgb_proba)
xgb_f1    = f1_score(y_test, xgb_pred)
xgb_rec   = recall_score(y_test, xgb_pred)
joblib.dump(xgb, 'model_xgboost.pkl')
print(f"AUC={xgb_auc:.3f} | F1={xgb_f1:.3f} | Rappel={xgb_rec:.3f} ✅")

# ── Sauvegarder les résultats ─────────────────────────────────────────
ml_results = {
    'decision_tree': {
        'auc': float(round(dt_auc, 3)), 'f1': float(round(dt_f1, 3)),
        'recall': float(round(dt_rec, 3)),
        'precision': float(round(precision_score(y_test, dt_pred), 3)),
        'accuracy':  float(round(accuracy_score(y_test, dt_pred), 3))
    },
    'random_forest': {
        'auc': float(round(rf_auc, 3)), 'f1': float(round(rf_f1, 3)),
        'recall': float(round(rf_rec, 3)),
        'precision': float(round(precision_score(y_test, rf_pred), 3)),
        'accuracy':  float(round(accuracy_score(y_test, rf_pred), 3))
    },
    'xgboost': {
        'auc': float(round(xgb_auc, 3)), 'f1': float(round(xgb_f1, 3)),
        'recall': float(round(xgb_rec, 3)),
        'precision': float(round(precision_score(y_test, xgb_pred), 3)),
        'accuracy':  float(round(accuracy_score(y_test, xgb_pred), 3))
    }
}
joblib.dump(ml_results, 'model_results.pkl')

# ── Importance des variables — Random Forest ──────────────────────────
feat_imp = pd.DataFrame({
    'Variable':   X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
feat_imp.to_csv('feature_importances.csv', index=False)

# ── Résumé final (comme le notebook section 7) ───────────────────────
aucs = {'Decision Tree': dt_auc, 'Random Forest': rf_auc, 'XGBoost': xgb_auc}
meilleur = max(aucs, key=aucs.get)

print("\n" + "=" * 60)
print("  TABLEAU COMPARATIF DES 3 ALGORITHMES")
print("=" * 60)
print(f"  {'Algorithme':<20} {'AUC':>6}  {'F1':>6}  {'Rappel':>7}")
print(f"  {'-'*20} {'-'*6}  {'-'*6}  {'-'*7}")
print(f"  {'Decision Tree':<20} {dt_auc:.3f}  {dt_f1:.3f}  {dt_rec:.3f}")
print(f"  {'Random Forest':<20} {rf_auc:.3f}  {rf_f1:.3f}  {rf_rec:.3f}")
print(f"  {'XGBoost':<20} {xgb_auc:.3f}  {xgb_f1:.3f}  {xgb_rec:.3f}")
print(f"\n  Meilleur modèle (AUC) : {meilleur} ({aucs[meilleur]:.3f})")
print("  Note : AUC 0.70-0.80 = acceptable en santé publique")

print("\n" + "=" * 60)
print("  SETUP TERMINÉ avec succès !")
print("=" * 60)
files_ok = ['bivariate_results.json', 'logit_results.json',
            'feature_columns.json', 'feature_importances.csv',
            'model_random_forest.pkl', 'model_decision_tree.pkl',
            'model_xgboost.pkl', 'model_results.pkl']
for f in files_ok:
    status = "✅" if os.path.exists(f) else "❌"
    print(f"   {status} {f}")
print("\n  Lancer l'application :")
print("  streamlit run app.py\n")
