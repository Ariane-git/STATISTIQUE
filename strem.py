"""
Application Streamlit — Facteurs associés à la Césarienne au Cameroun
EDS 2018 — Tous les calculs sont faits au démarrage, aucun fichier JSON requis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Césarienne — EDS Cameroun 2018",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a5276, #2e86ab);
    color: white; padding: 1.5rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem; text-align: center;
}
.metric-card {
    background: #f8f9fa; border-left: 4px solid #2e86ab;
    padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
}
.prediction-box {
    padding: 1.5rem; border-radius: 12px; text-align: center;
    font-size: 1.2rem; font-weight: bold; margin: 1rem 0;
}
.pred-high { background:#fadbd8; border:2px solid #e74c3c; color:#c0392b; }
.pred-low  { background:#d5f5e3; border:2px solid #27ae60; color:#1e8449; }
.pred-mod  { background:#fef9e7; border:2px solid #f39c12; color:#d68910; }
</style>
""", unsafe_allow_html=True)

# ─── Chargement et prétraitement des données ──────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("data_cesarienne_complet1.csv", encoding='utf-8')

    # Valeurs manquantes
    df['number_of_antenatal_visits_during_p_m14'] = pd.to_numeric(
        df['number_of_antenatal_visits_during_p_m14'], errors='coerce')
    df['number_of_antenatal_visits_during_p_m14'].fillna(
        df['number_of_antenatal_visits_during_p_m14'].mode()[0], inplace=True)

    # Variable cible
    df['cesarienne'] = df['delivery_by_caesarean_section_m17'].map({'yes': 1, 'no': 0})

    # Age
    df['age_groupe'] = pd.cut(df['Respondent_s_current_age_v012'],
        bins=[0,19,34,99], labels=['Moins de 20 ans','20-34 ans','35 ans et plus']
    ).astype(str)

    # Milieu
    df['milieu'] = df['Type_of_place_of_residence_v025'].map(
        {'Urban': 'Urbain', 'Rural': 'Rural'})

    # Instruction
    df['instruction'] = df['Highest_educational_level_v106'].map({
        'No education': 'Aucune', 'Primary': 'Primaire',
        'Secondary': 'Secondaire', 'Higher': 'Superieur'})

    # Richesse
    df['richesse'] = df['Wealth_index_combined_v190'].map({
        'Poorest': 'Tres pauvre', 'Poorer': 'Pauvre', 'Middle': 'Moyen',
        'Richer': 'Riche', 'Richest': 'Tres riche'})

    # Parite
    df['parite'] = np.where(df['Total_children_ever_born_v201'] == 1,
                             'Primipare', 'Multipare')

    # Statut
    df['statut_union'] = np.where(
        df['Current_marital_status_v501'].isin(['Married', 'Living with partner']),
        'En union', 'Pas en union')

    # Sexe
    df['sexe_enfant'] = df['sex_of_child_b4'].map(
        {'male': 'Masculin', 'female': 'Feminin'})

    # CVA
    df['visites_pn'] = pd.to_numeric(
        df['number_of_antenatal_visits_during_p_m14'], errors='coerce')
    df['cva'] = np.where(df['visites_pn'] >= 4,
                          '4 visites et plus', 'Moins de 4 visites')

    # Region
    df['region'] = df['Region_v024'].astype(str)

    # Lieu accouchement
    def classer_lieu(lieu):
        if pd.isna(lieu): return 'Inconnu'
        lieu = str(lieu).lower()
        if 'home' in lieu:                                  return 'Domicile'
        elif 'government' in lieu or 'public' in lieu:     return 'Secteur public'
        elif 'integrated' in lieu or 'dispensary' in lieu: return 'Secteur public'
        elif 'confessional' in lieu:                        return 'Confessionnel'
        elif 'private' in lieu or 'clinic' in lieu:        return 'Secteur prive'
        else:                                               return 'Autre'
    df['lieu_accouchement'] = df['place_of_delivery_m15'].apply(classer_lieu)

    # Grossesse desiree
    df['grossesse_desiree'] = df['Wanted_last_child_v367'].map({
        'Wanted then': 'Desiree alors',
        'Wanted later': 'Desiree plus tard',
        'Wanted no more': 'Non desiree'})

    return df.dropna(subset=['cesarienne'])


@st.cache_data
def compute_bivariate(_df):
    """Calcule les analyses bivariees directement depuis le dataframe."""
    from scipy.stats import chi2_contingency

    vars_biv = {
        'age_groupe':        "Groupe d'age",
        'milieu':            'Milieu de residence',
        'instruction':       "Niveau d'instruction",
        'richesse':          'Indice de richesse',
        'parite':            'Parite',
        'statut_union':      'Statut matrimonial',
        'sexe_enfant':       "Sexe de l'enfant",
        'cva':               'Visites prenatales',
        'lieu_accouchement': "Lieu d'accouchement",
        'region':            'Region'
    }

    results = {}
    for var, label in vars_biv.items():
        sub = _df[[var, 'cesarienne']].dropna()
        ct     = pd.crosstab(sub[var], sub['cesarienne'])
        ct_pct = pd.crosstab(sub[var], sub['cesarienne'], normalize='index') * 100
        chi2, p, _, _ = chi2_contingency(ct)
        tab_data = {}
        for mod in ct.index:
            tab_data[str(mod)] = {
                'n':       int(ct.loc[mod].sum()),
                'n_ces':   int(ct.loc[mod, 1]) if 1 in ct.columns else 0,
                'pct_ces': round(float(ct_pct.loc[mod, 1]) if 1 in ct_pct.columns else 0.0, 1)
            }
        results[var] = {
            'label':   label,
            'p_value': round(float(p), 4),
            'chi2':    round(float(chi2), 2),
            'data':    tab_data
        }
    return results


@st.cache_data
def compute_logit(_df):
    """Calcule la regression logistique multivariee."""
    import statsmodels.api as sm

    vars_logit = ['age_groupe', 'milieu', 'instruction', 'richesse',
                  'parite', 'statut_union', 'sexe_enfant', 'cva']
    sub = _df[vars_logit + ['cesarienne']].dropna()
    X   = pd.get_dummies(sub[vars_logit], drop_first=True).astype(float)
    Xc  = sm.add_constant(X)
    y   = sub['cesarienne'].astype(float)

    result = sm.Logit(y, Xc).fit(disp=0)

    rows = []
    for var_name in result.params.index:
        coef    = float(result.params[var_name])
        pval    = float(result.pvalues[var_name])
        ci_low  = float(result.conf_int().loc[var_name, 0])
        ci_high = float(result.conf_int().loc[var_name, 1])
        rows.append({
            'variable':    str(var_name),
            'OR':          round(np.exp(coef), 3),
            'CI_low':      round(np.exp(ci_low), 3),
            'CI_high':     round(np.exp(ci_high), 3),
            'p_value':     round(pval, 4),
            'significant': bool(pval < 0.05)
        })
    return rows


@st.cache_resource
def load_models():
    models = {}
    for name in ['random_forest', 'decision_tree', 'xgboost']:
        path = f'model_{name}.pkl'
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_data
def compute_features(_df):
    """Prépare les features ML (labels sans accents, identique au notebook ML)."""
    df2 = _df.copy()
    df2['age_ml'] = pd.cut(df2['Respondent_s_current_age_v012'],
        bins=[0,19,34,99], labels=['Moins_20','20_34','35_plus']).astype(str)
    df2['mil_ml']  = df2['Type_of_place_of_residence_v025'].map({'Urban':'Urbain','Rural':'Rural'})
    df2['ins_ml']  = df2['Highest_educational_level_v106'].map({
        'No education':'Aucune','Primary':'Primaire',
        'Secondary':'Secondaire','Higher':'Superieur'})
    df2['ric_ml']  = df2['Wealth_index_combined_v190'].map({
        'Poorest':'Tres_pauvre','Poorer':'Pauvre','Middle':'Moyen',
        'Richer':'Riche','Richest':'Tres_riche'})
    df2['par_ml']  = np.where(df2['Total_children_ever_born_v201']==1,'Primipare','Multipare')
    df2['stat_ml'] = np.where(
        df2['Current_marital_status_v501'].isin(['Married','Living with partner']),
        'En_union','Pas_en_union')
    df2['sex_ml']  = df2['sex_of_child_b4'].map({'male':'Masculin','female':'Feminin'})
    df2['cva_ml']  = np.where(df2['visites_pn'] >= 4, '4_plus', 'Moins_4')
    df2['reg_ml']  = df2['Region_v024'].astype(str)

    cols = ['age_ml','mil_ml','ins_ml','ric_ml','par_ml','stat_ml','sex_ml','cva_ml','reg_ml']
    sub  = df2[cols + ['cesarienne']].dropna()
    X    = pd.get_dummies(sub[cols], drop_first=False, dtype=int)
    X.columns = [
        c.replace('age_ml_','age_groupe_').replace('mil_ml_','milieu_')
         .replace('ins_ml_','instruction_').replace('ric_ml_','richesse_')
         .replace('par_ml_','parite_').replace('stat_ml_','statut_union_')
         .replace('sex_ml_','sexe_enfant_').replace('cva_ml_','cva_')
         .replace('reg_ml_','region_')
        for c in X.columns
    ]
    return X, sub['cesarienne'].astype(int)


@st.cache_data
def compute_feature_importance(_df):
    models = load_models()
    if 'random_forest' not in models:
        return pd.DataFrame()
    X, _ = compute_features(_df)
    X  = X.copy()
    rf = models['random_forest']
    fi   = pd.DataFrame({'Variable': X.columns,
                          'Importance': rf.feature_importances_})
    return fi.sort_values('Importance', ascending=False).reset_index(drop=True)


@st.cache_data
def compute_ml_results(_df):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

    models = load_models()
    if not models:
        return {}

    X, y = compute_features(_df)
    # Copie explicite pour eviter le flag WRITEABLE=False de st.cache_data
    X = X.copy()
    y = y.copy()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20,
                                             random_state=42, stratify=y)
    results = {}
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred  = model.predict(X_test)
        results[name] = {
            'auc':       round(float(roc_auc_score(y_test, proba)), 3),
            'f1':        round(float(f1_score(y_test, pred)), 3),
            'recall':    round(float(recall_score(y_test, pred)), 3),
            'precision': round(float(precision_score(y_test, pred)), 3),
            'accuracy':  round(float(accuracy_score(y_test, pred)), 3),
        }
    return results


# ─── Chargement ───────────────────────────────────────────────────────────────
df = load_and_prepare()

with st.spinner("Calcul des analyses statistiques..."):
    biv_results  = compute_bivariate(df)
    logit_results = compute_logit(df)
    models       = load_models()
    feat_imp     = compute_feature_importance(df)
    ml_results   = compute_ml_results(df)

X_feat, _ = compute_features(df)
feat_cols  = X_feat.columns.tolist()

# ─── Constantes ───────────────────────────────────────────────────────────────
n_ces   = int(df['cesarienne'].sum())
n_total = len(df)
taux    = df['cesarienne'].mean() * 100
model_names = {
    'random_forest': 'Random Forest',
    'decision_tree': 'Arbre de Decision',
    'xgboost':       'XGBoost'
}
model_colors = {
    'random_forest': '#27ae60',
    'decision_tree': '#2e86ab',
    'xgboost':       '#e74c3c'
}

# ─── En-tete ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem;">
        Facteurs associes a la Cesarienne au Cameroun
    </h1>
    <p style="margin:0.5rem 0 0;font-size:1rem;opacity:0.9;">
        Enquete Demographique et de Sante (EDS) 2018 &nbsp;|&nbsp;
        <b>n = {n_total:,} naissances</b>
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total naissances",       f"{n_total:,}")
col2.metric("Cesariennes",            f"{n_ces:,}",           f"{taux:.1f}%")
col3.metric("Accouchements vaginaux", f"{n_total - n_ces:,}", f"{100-taux:.1f}%")
col4.metric("Regions couvertes",      df['region'].nunique())
st.markdown("---")

# ─── Onglets ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Analyse descriptive",
    "Analyse bivariee",
    "Regression logistique",
    "Machine Learning",
    "Prediction individuelle"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIF
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Distribution de la variable dependante")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(
            labels=['Voie basse', 'Cesarienne'],
            values=[n_total - n_ces, n_ces],
            hole=0.4,
            marker_colors=['#2e86ab','#e74c3c'],
            textinfo='label+percent', textfont_size=13
        ))
        fig.update_layout(
            title="Mode d'accouchement", height=380, showlegend=False,
            annotations=[dict(text=f"{taux:.1f}%<br>C-section",
                               x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        reg = df.groupby('region')['cesarienne'].agg(['mean','count']).reset_index()
        reg.columns = ['region','taux','n']
        reg['pct'] = (reg['taux']*100).round(1)
        reg = reg.sort_values('pct')
        fig2 = px.bar(reg, x='pct', y='region', orientation='h',
            text='pct', color='pct', color_continuous_scale='RdYlGn_r',
            title="Taux de cesarienne par region (%)",
            labels={'pct':'Taux (%)','region':'Region'})
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Distribution des variables explicatives")
    vars_desc = {
        'age_groupe': "Groupe d'age", 'milieu': 'Milieu',
        'instruction': 'Instruction', 'richesse': 'Richesse',
        'parite': 'Parite', 'statut_union': 'Statut matrimonial',
        'sexe_enfant': 'Sexe enfant', 'cva': 'Visites prenatales',
        'lieu_accouchement': "Lieu d'accouchement"
    }
    var_c = st.selectbox("Variable", list(vars_desc.keys()),
                          format_func=lambda x: vars_desc[x])
    ct     = pd.crosstab(df[var_c], df['cesarienne'])
    ct_pct = pd.crosstab(df[var_c], df['cesarienne'], normalize='index')*100

    plot_df = pd.DataFrame({
        'Modalite': ct.index.astype(str),
        'Voie basse': ct_pct[0].round(1) if 0 in ct_pct.columns else [0]*len(ct),
        'Cesarienne': ct_pct[1].round(1) if 1 in ct_pct.columns else [0]*len(ct),
    })
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name='Voie basse', x=plot_df['Modalite'],
                           y=plot_df['Voie basse'], marker_color='#2e86ab'))
    fig3.add_trace(go.Bar(name='Cesarienne', x=plot_df['Modalite'],
                           y=plot_df['Cesarienne'], marker_color='#e74c3c'))
    fig3.update_layout(barmode='stack', height=380,
        title=f"Repartition par {vars_desc[var_c]}",
        yaxis_title="%",
        legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — BIVARIEE
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Association Chi2 entre chaque facteur et la cesarienne")

    rows = []
    for var, info in biv_results.items():
        p   = info['p_value']
        sig = "Significatif" if p < 0.05 else "NS"
        for mod, dat in info['data'].items():
            rows.append({
                'Variable':       info['label'],
                'Modalite':       mod,
                'n':              dat['n'],
                'n cesarienne':   dat['n_ces'],
                'Taux (%)':       dat['pct_ces'],
                'p Chi2':         p,
                'Association':    sig
            })
    df_biv = pd.DataFrame(rows)

    filtre = st.selectbox("Filtrer", ["Toutes","Significatives","Non significatives"])
    if filtre == "Significatives":
        df_biv = df_biv[df_biv['Association'] == "Significatif"]
    elif filtre == "Non significatives":
        df_biv = df_biv[df_biv['Association'] == "NS"]

    st.dataframe(
        df_biv.style
              .background_gradient(subset=['Taux (%)'], cmap='Reds')
              .format({'Taux (%)': '{:.1f}', 'p Chi2': '{:.4f}'}),
        use_container_width=True, height=420
    )

    st.subheader("Taux de cesarienne par variable")
    var_biv = st.selectbox("Variable", list(biv_results.keys()),
                            format_func=lambda x: biv_results[x]['label'])
    d = biv_results[var_biv]
    mods  = list(d['data'].keys())
    taux_v = [d['data'][m]['pct_ces'] for m in mods]
    fig4 = px.bar(x=mods, y=taux_v, text=taux_v, color=taux_v,
        color_continuous_scale='Reds',
        title=f"Taux de cesarienne — {d['label']} (p={d['p_value']:.4f})",
        labels={'x': d['label'], 'y': 'Taux (%)'})
    fig4.add_hline(y=taux, line_dash="dash", line_color="#2e86ab",
                   annotation_text=f"Moyenne: {taux:.1f}%")
    fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig4.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — REGRESSION LOGISTIQUE
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Regression logistique multivariee — Odds Ratios ajustes")
    st.caption("Reference : 20-34 ans | Rural | Aucune instruction | "
               "Tres pauvre | Multipare | En union | Feminin | 4 visites et plus")

    df_logit = pd.DataFrame(logit_results)
    df_logit = df_logit[df_logit['variable'] != 'const'].copy()

    c1, c2 = st.columns([3, 2])
    with c1:
        df_s = df_logit.sort_values('OR', ascending=True)
        fig5 = go.Figure()
        for _, row in df_s.iterrows():
            color = '#e74c3c' if row['significant'] else '#95a5a6'
            fig5.add_trace(go.Scatter(
                x=[row['CI_low'], row['OR'], row['CI_high']],
                y=[row['variable']]*3,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=[5,10,5], color=[color,color,color]),
                showlegend=False
            ))
        fig5.add_vline(x=1, line_dash="dash", line_color="#2e86ab", line_width=2)
        fig5.update_layout(
            title="Forest Plot — Odds Ratios (IC 95%)",
            xaxis_title="Odds Ratio", xaxis_type='log', height=500,
            yaxis=dict(tickfont=dict(size=10))
        )
        st.plotly_chart(fig5, use_container_width=True)

    with c2:
        st.markdown("**Tableau des resultats**")
        df_d = df_logit[['variable','OR','CI_low','CI_high','p_value','significant']].copy()
        df_d.columns = ['Variable','OR','IC 2.5%','IC 97.5%','p-valeur','Sig.']
        df_d['Sig.'] = df_d['Sig.'].map({True: 'OUI', False: 'non'})
        st.dataframe(
            df_d.style.format({'OR':'{:.2f}','IC 2.5%':'{:.2f}',
                               'IC 97.5%':'{:.2f}','p-valeur':'{:.4f}'}),
            use_container_width=True, height=420
        )

    sig_f = df_logit[df_logit['significant']].sort_values('OR', ascending=False)
    st.subheader("Facteurs significativement associes a la cesarienne")
    cols_s = st.columns(3)
    for i, (_, row) in enumerate(sig_f.iterrows()):
        with cols_s[i % 3]:
            direction = "Augmente" if row['OR'] > 1 else "Diminue"
            st.markdown(f"""
            <div class="metric-card">
                <b>{row['variable']}</b><br>
                OR = {row['OR']:.2f} [{row['CI_low']:.2f}–{row['CI_high']:.2f}]<br>
                {direction} le risque<br>
                <small>p = {row['p_value']:.4f}</small>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Comparaison des 3 algorithmes de Machine Learning")

    if ml_results:
        metrics_data = []
        for key, name in model_names.items():
            if key in ml_results:
                r = ml_results[key]
                metrics_data.append({
                    'Modele': name, 'AUC-ROC': r['auc'], 'F1-Score': r['f1'],
                    'Rappel': r['recall'], 'Precision': r['precision'],
                    'Exactitude': r['accuracy']
                })
        df_m = pd.DataFrame(metrics_data)

        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("**Performance des modeles**")
            st.dataframe(
                df_m.style.highlight_max(
                    subset=['AUC-ROC','F1-Score','Rappel','Precision','Exactitude'],
                    color='#d5f5e3'
                ).format({'AUC-ROC':'{:.3f}','F1-Score':'{:.3f}',
                          'Rappel':'{:.3f}','Precision':'{:.3f}','Exactitude':'{:.3f}'}),
                use_container_width=True
            )
            best = df_m.loc[df_m['AUC-ROC'].idxmax(), 'Modele']
            st.success(f"Meilleur modele (AUC) : **{best}**")

        with c2:
            cats = ['AUC-ROC','F1-Score','Rappel','Precision','Exactitude']
            fig6 = go.Figure()
            for _, row in df_m.iterrows():
                key = [k for k,v in model_names.items() if v == row['Modele']][0]
                fig6.add_trace(go.Scatterpolar(
                    r=[row[c] for c in cats], theta=cats, fill='toself',
                    name=row['Modele'], line_color=model_colors[key], opacity=0.6
                ))
            fig6.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                title="Comparaison des metriques", height=380
            )
            st.plotly_chart(fig6, use_container_width=True)
    else:
        st.warning("Modeles ML non trouves. Lancez setup.py d'abord.")

    if not feat_imp.empty:
        st.subheader("Importance des variables — Random Forest")
        top_n = st.slider("Nombre de variables", 10, len(feat_imp), 20)
        fi_top = feat_imp.head(top_n).sort_values('Importance')
        fig7 = px.bar(fi_top, x='Importance', y='Variable', orientation='h',
            color='Importance', color_continuous_scale='Blues',
            title=f"Top {top_n} variables importantes",
            labels={'Importance': "Score d'importance (Gini)", 'Variable': ''})
        fig7.update_layout(height=max(400, top_n*22), coloraxis_showscale=False)
        st.plotly_chart(fig7, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICTION
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Prediction du risque de cesarienne pour un profil individuel")
    st.info("Outil a visee pedagogique et de recherche — pas un outil clinique.")

    if not models:
        st.error("Aucun modele trouve. Lancez setup.py d'abord.")
    else:
        c_form, c_res = st.columns([1, 1])
        with c_form:
            st.markdown("**Caracteristiques de la femme**")
            age_i   = st.selectbox("Groupe d'age",
                ['Moins_20','20_34','35_plus'],
                format_func=lambda x: {'Moins_20':'< 20 ans','20_34':'20-34 ans','35_plus':'>= 35 ans'}[x])
            mil_i   = st.selectbox("Milieu de residence", ['Urbain','Rural'])
            ins_i   = st.selectbox("Niveau d'instruction",
                ['Aucune','Primaire','Secondaire','Superieur'])
            ric_i   = st.selectbox("Indice de richesse",
                ['Tres_pauvre','Pauvre','Moyen','Riche','Tres_riche'],
                format_func=lambda x: x.replace('_',' '))
            par_i   = st.selectbox("Parite", ['Primipare','Multipare'])
            stat_i  = st.selectbox("Statut matrimonial",
                ['En_union','Pas_en_union'],
                format_func=lambda x: x.replace('_',' '))
            sex_i   = st.selectbox("Sexe de l'enfant", ['Masculin','Feminin'])
            cva_i   = st.selectbox("Visites prenatales",
                ['4_plus','Moins_4'],
                format_func=lambda x: {'4_plus':'4 visites et plus','Moins_4':'Moins de 4 visites'}[x])
            reg_i   = st.selectbox("Region", sorted(df['region'].unique()))
            mod_c   = st.radio("Modele ML",
                list(models.keys()),
                format_func=lambda x: model_names.get(x, x),
                horizontal=True)
            btn = st.button("Calculer le risque", type="primary", use_container_width=True)

        with c_res:
            if btn:
                input_dict = {c: 0 for c in feat_cols}
                for k in [f'age_groupe_{age_i}', f'milieu_{mil_i}',
                           f'instruction_{ins_i}', f'richesse_{ric_i}',
                           f'parite_{par_i}', f'statut_union_{stat_i}',
                           f'sexe_enfant_{sex_i}', f'cva_{cva_i}',
                           f'region_{reg_i}']:
                    if k in input_dict:
                        input_dict[k] = 1

                X_pred = pd.DataFrame([input_dict])[feat_cols]
                proba  = models[mod_c].predict_proba(X_pred)[0][1]
                pct    = proba * 100

                if pct >= 20:
                    css, icon, niv = "pred-high", "🔴", "RISQUE ELEVE"
                elif pct >= 10:
                    css, icon, niv = "pred-mod",  "🟡", "RISQUE MODERE"
                else:
                    css, icon, niv = "pred-low",  "🟢", "RISQUE FAIBLE"

                st.markdown(f"""
                <div class="prediction-box {css}">
                    {icon} {niv}<br>
                    <span style="font-size:2.5rem;">{pct:.1f}%</span><br>
                    <span style="font-size:0.9rem;font-weight:normal;">
                        probabilite estimee de cesarienne
                    </span>
                </div>
                """, unsafe_allow_html=True)

                fig8 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pct,
                    number={'suffix':'%','font':{'size':30}},
                    gauge={
                        'axis': {'range':[0,50],'ticksuffix':'%'},
                        'bar':  {'color':'#e74c3c' if pct>=20 else '#f39c12' if pct>=10 else '#27ae60'},
                        'steps': [
                            {'range':[0,10],  'color':'#d5f5e3'},
                            {'range':[10,20], 'color':'#fef9e7'},
                            {'range':[20,50], 'color':'#fadbd8'}
                        ],
                        'threshold': {'line':{'color':'#2e86ab','width':3},
                                      'thickness':0.75,'value':taux}
                    },
                    title={'text': f"Probabilite estimee<br>"
                                   f"<sup>Modele : {model_names[mod_c]}</sup>"}
                ))
                fig8.update_layout(height=280)
                st.plotly_chart(fig8, use_container_width=True)
                st.info(f"Taux national : **{taux:.1f}%** — Ce profil : "
                        f"**{pct:.1f}%** ({pct/taux:.1f}x la moyenne)")
            else:
                st.markdown("""
                <div style='text-align:center;color:#7f8c8d;padding:3rem 1rem;'>
                    <div style='font-size:3rem;'>🔮</div>
                    <p>Renseignez les caracteristiques puis cliquez sur<br>
                    <b>Calculer le risque</b></p>
                </div>
                """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"EDS Cameroun 2018 — n={n_total:,} — Usage academique uniquement")