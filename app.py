"""
Cesarienne EDS Cameroun 2018 — Application Streamlit
Navbar verticale dans le sidebar, calculs 100% en memoire.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, os, warnings
warnings.filterwarnings('ignore')

# Répertoire absolu du fichier app.py — fonctionne quel que soit le cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cesarienne — EDS Cameroun 2018",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Navbar boutons sidebar */
.nav-btn {
    display: block; width: 100%; text-align: left;
    padding: 10px 16px; margin: 4px 0;
    border-radius: 8px; border: none;
    font-size: 0.95rem; font-weight: 500;
    cursor: pointer; transition: background 0.2s;
}
.nav-btn-active {
    background: #2e86ab; color: white !important;
}
.nav-btn-inactive {
    background: transparent; color: inherit;
}
.nav-btn-inactive:hover { background: rgba(46,134,171,0.15); }

/* Cards facteurs significatifs */
.fact-card {
    background: #1e3a4a;
    border-left: 4px solid #2e86ab;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.6;
}
.fact-card-red  { border-left-color: #e74c3c; background: #3a1e1e; }
.fact-card-green{ border-left-color: #27ae60; background: #1e3a28; }

/* Prediction box */
.pred-box {
    padding: 1.4rem; border-radius: 12px;
    text-align: center; font-weight: bold;
    margin: 0.8rem 0; font-size: 1.1rem;
}
.pred-high { background:#4a1a1a; border:2px solid #e74c3c; color:#ff6b6b; }
.pred-mod  { background:#3a3010; border:2px solid #f39c12; color:#f5c842; }
.pred-low  { background:#1a3a2a; border:2px solid #27ae60; color:#58d68d; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #1a5276, #2e86ab);
    color: white; padding: 1.2rem 1.5rem; border-radius: 10px;
    margin-bottom: 1rem; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT DONNEES
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data_cesarienne_complet1.csv"), encoding="utf-8")
    df["number_of_antenatal_visits_during_p_m14"] = pd.to_numeric(
        df["number_of_antenatal_visits_during_p_m14"], errors="coerce")
    df["number_of_antenatal_visits_during_p_m14"].fillna(
        df["number_of_antenatal_visits_during_p_m14"].mode()[0], inplace=True)
    df["cesarienne"] = df["delivery_by_caesarean_section_m17"].map({"yes":1,"no":0})
    df["visites_pn"] = pd.to_numeric(
        df["number_of_antenatal_visits_during_p_m14"], errors="coerce")

    df["age_groupe"]   = pd.cut(df["Respondent_s_current_age_v012"],
        bins=[0,19,34,99], labels=["Moins 20 ans","20-34 ans","35 ans et plus"]).astype(str)
    df["milieu"]       = df["Type_of_place_of_residence_v025"].map({"Urban":"Urbain","Rural":"Rural"})
    df["instruction"]  = df["Highest_educational_level_v106"].map({
        "No education":"Aucune","Primary":"Primaire",
        "Secondary":"Secondaire","Higher":"Superieur"})
    df["richesse"]     = df["Wealth_index_combined_v190"].map({
        "Poorest":"Tres pauvre","Poorer":"Pauvre","Middle":"Moyen",
        "Richer":"Riche","Richest":"Tres riche"})
    df["parite"]       = np.where(df["Total_children_ever_born_v201"]==1,"Primipare","Multipare")
    df["statut_union"] = np.where(
        df["Current_marital_status_v501"].isin(["Married","Living with partner"]),
        "En union","Pas en union")
    df["sexe_enfant"]  = df["sex_of_child_b4"].map({"male":"Masculin","female":"Feminin"})
    df["cva"]          = np.where(df["visites_pn"]>=4,"4 visites et plus","Moins de 4 visites")
    df["region"]       = df["Region_v024"].astype(str)

    def classer_lieu(l):
        if pd.isna(l): return "Inconnu"
        l = str(l).lower()
        if "home"        in l: return "Domicile"
        if "government"  in l or "public"     in l: return "Secteur public"
        if "dispensary"  in l or "integrated" in l: return "Secteur public"
        if "confessional"in l: return "Confessionnel"
        if "private"     in l or "clinic"     in l: return "Secteur prive"
        return "Autre"
    df["lieu_accouchement"] = df["place_of_delivery_m15"].apply(classer_lieu)
    df["grossesse_desiree"] = df["Wanted_last_child_v367"].map({
        "Wanted then":"Desiree alors",
        "Wanted later":"Desiree plus tard",
        "Wanted no more":"Non desiree"})
    return df.dropna(subset=["cesarienne"])


@st.cache_data
def compute_bivariate(_df):
    from scipy.stats import chi2_contingency
    vars_biv = {
        "age_groupe":"Groupe d age","milieu":"Milieu de residence",
        "instruction":"Niveau d instruction","richesse":"Indice de richesse",
        "parite":"Parite","statut_union":"Statut matrimonial",
        "sexe_enfant":"Sexe de l enfant","cva":"Visites prenatales",
        "lieu_accouchement":"Lieu d accouchement","region":"Region"
    }
    res = {}
    for var, label in vars_biv.items():
        sub = _df[[var,"cesarienne"]].dropna()
        ct     = pd.crosstab(sub[var], sub["cesarienne"])
        ct_pct = pd.crosstab(sub[var], sub["cesarienne"], normalize="index")*100
        chi2, p, _, _ = chi2_contingency(ct)
        tab = {}
        for mod in ct.index:
            tab[str(mod)] = {
                "n":       int(ct.loc[mod].sum()),
                "n_ces":   int(ct.loc[mod,1]) if 1 in ct.columns else 0,
                "pct_ces": round(float(ct_pct.loc[mod,1]) if 1 in ct_pct.columns else 0.0, 1)
            }
        res[var] = {"label":label, "p_value":round(float(p),4),
                    "chi2":round(float(chi2),2), "data":tab}
    return res


@st.cache_data
def compute_logit(_df):
    import statsmodels.api as sm
    vars_l = ["age_groupe","milieu","instruction","richesse",
              "parite","statut_union","sexe_enfant","cva"]
    sub = _df[vars_l+["cesarienne"]].dropna()
    X   = pd.get_dummies(sub[vars_l], drop_first=True).astype(float)
    Xc  = sm.add_constant(X)
    y   = sub["cesarienne"].astype(float)
    r   = sm.Logit(y, Xc).fit(disp=0)
    rows = []
    for v in r.params.index:
        c = float(r.params[v]); p = float(r.pvalues[v])
        lo = float(r.conf_int().loc[v,0]); hi = float(r.conf_int().loc[v,1])
        rows.append({"variable":str(v), "OR":round(np.exp(c),3),
                     "CI_low":round(np.exp(lo),3), "CI_high":round(np.exp(hi),3),
                     "p_value":round(p,4), "significant":bool(p<0.05)})
    return rows


@st.cache_data
def compute_features(_df):
    df2 = _df.copy()
    df2["age_ml"]  = pd.cut(df2["Respondent_s_current_age_v012"],
        bins=[0,19,34,99], labels=["Moins_20","20_34","35_plus"]).astype(str)
    df2["mil_ml"]  = df2["Type_of_place_of_residence_v025"].map({"Urban":"Urbain","Rural":"Rural"})
    df2["ins_ml"]  = df2["Highest_educational_level_v106"].map({
        "No education":"Aucune","Primary":"Primaire",
        "Secondary":"Secondaire","Higher":"Superieur"})
    df2["ric_ml"]  = df2["Wealth_index_combined_v190"].map({
        "Poorest":"Tres_pauvre","Poorer":"Pauvre","Middle":"Moyen",
        "Richer":"Riche","Richest":"Tres_riche"})
    df2["par_ml"]  = np.where(df2["Total_children_ever_born_v201"]==1,"Primipare","Multipare")
    df2["stat_ml"] = np.where(
        df2["Current_marital_status_v501"].isin(["Married","Living with partner"]),
        "En_union","Pas_en_union")
    df2["sex_ml"]  = df2["sex_of_child_b4"].map({"male":"Masculin","female":"Feminin"})
    df2["cva_ml"]  = np.where(df2["visites_pn"]>=4,"4_plus","Moins_4")
    df2["reg_ml"]  = df2["Region_v024"].astype(str)
    cols = ["age_ml","mil_ml","ins_ml","ric_ml","par_ml","stat_ml","sex_ml","cva_ml","reg_ml"]
    sub  = df2[cols+["cesarienne"]].dropna()
    X    = pd.get_dummies(sub[cols], drop_first=False, dtype=int)
    X.columns = [c.replace("age_ml_","age_groupe_").replace("mil_ml_","milieu_")
                  .replace("ins_ml_","instruction_").replace("ric_ml_","richesse_")
                  .replace("par_ml_","parite_").replace("stat_ml_","statut_union_")
                  .replace("sex_ml_","sexe_enfant_").replace("cva_ml_","cva_")
                  .replace("reg_ml_","region_") for c in X.columns]
    return X, sub["cesarienne"].astype(int)


@st.cache_resource
def load_models():
    """Charge les 6 modeles ML depuis les fichiers .pkl."""
    ALL_MODELS = [
        "logistic_regression", "decision_tree", "random_forest",
        "xgboost", "gradient_boosting", "knn"
    ]
    m = {}
    for name in ALL_MODELS:
        p = os.path.join(BASE_DIR, f"model_{name}.pkl")
        if os.path.exists(p):
            m[name] = joblib.load(p)
    return m


@st.cache_resource
def load_scaler():
    """Charge le scaler pour les modeles qui en ont besoin (KNN, LR)."""
    if os.path.exists(os.path.join(BASE_DIR, "scaler.pkl")):
        return joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return None


@st.cache_data
def load_ml_results():
    """Charge les metriques pre-calculees par setup.py."""
    if os.path.exists(os.path.join(BASE_DIR, "model_results.pkl")):
        return joblib.load(os.path.join(BASE_DIR, "model_results.pkl"))
    return {}


@st.cache_data
def load_shap_results():
    """Charge les valeurs SHAP pre-calculees par setup.py."""
    if os.path.exists(os.path.join(BASE_DIR, "shap_results.pkl")):
        return joblib.load(os.path.join(BASE_DIR, "shap_results.pkl"))
    return None


@st.cache_data
def compute_feat_importance(_df):
    models = load_models()
    if "random_forest" not in models:
        return pd.DataFrame()
    X, _ = compute_features(_df)
    X = X.copy()
    fi = pd.DataFrame({"Variable": X.columns,
                        "Importance": models["random_forest"].feature_importances_})
    return fi.sort_values("Importance", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════
df          = load_data()
biv         = compute_bivariate(df)
logit_rows  = compute_logit(df)
models      = load_models()
scaler      = load_scaler()
ml_metrics  = load_ml_results()
shap_data   = load_shap_results()
feat_imp    = compute_feat_importance(df)
X_feat, _   = compute_features(df)
feat_cols   = X_feat.columns.tolist()

n_ces   = int(df["cesarienne"].sum())
n_total = len(df)
taux    = df["cesarienne"].mean() * 100

MODEL_NAMES = {
    "logistic_regression": "Regression Logistique",
    "decision_tree":       "Arbre de Decision",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "gradient_boosting":   "Gradient Boosting",
    "knn":                 "K-Nearest Neighbors",
}
MODEL_COLORS = {
    "logistic_regression": "#8e44ad",
    "decision_tree":       "#2e86ab",
    "random_forest":       "#27ae60",
    "xgboost":             "#e74c3c",
    "gradient_boosting":   "#e67e22",
    "knn":                 "#16a085",
}
# Modeles necessitant le scaler pour la prediction
SCALED_MODELS = {"logistic_regression", "knn"}

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — NAVBAR VERTICALE
# ══════════════════════════════════════════════════════════════════════
PAGES = [
    ("📊", "Analyse descriptive"),
    ("🔬", "Analyse bivariee"),
    ("📈", "Regression logistique"),
    ("🤖", "Machine Learning"),
    ("🧠", "Analyse SHAP"),
    ("🔮", "Prediction"),
]

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.5rem 0 1rem 0;'>
        <div style='font-size:2rem;'>🏥</div>
        <div style='font-weight:700; font-size:1rem; line-height:1.3;'>
            Cesarienne<br>EDS Cameroun 2018
        </div>
        <div style='font-size:0.75rem; opacity:0.7; margin-top:4px;'>
            n = {:,} naissances
        </div>
    </div>
    """.format(n_total), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Navigation**")

    if "page" not in st.session_state:
        st.session_state.page = "Analyse descriptive"

    for icon, label in PAGES:
        is_active = st.session_state.page == label
        btn_style = (
            "background:#2e86ab;color:white;border:none;width:100%;"
            "text-align:left;padding:10px 14px;border-radius:8px;"
            "margin:3px 0;font-size:0.92rem;font-weight:600;cursor:pointer;"
        ) if is_active else (
            "background:transparent;border:none;width:100%;"
            "text-align:left;padding:10px 14px;border-radius:8px;"
            "margin:3px 0;font-size:0.92rem;cursor:pointer;"
        )
        if st.button(f"{icon}  {label}", key=f"nav_{label}",
                     use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = label
            st.rerun()

    st.markdown("---")
    # Mini stats dans sidebar
    st.markdown("**Statistiques globales**")
    st.metric("Taux cesarienne", f"{taux:.1f}%")
    st.metric("Modeles ML charges", f"{len(models)}/6")
    if ml_metrics:
        best_name = max(ml_metrics, key=lambda k: ml_metrics[k]["auc"])
        best_auc  = ml_metrics[best_name]["auc"]
        st.metric("Meilleur AUC", f"{best_auc:.3f}")
        st.caption(f"({MODEL_NAMES.get(best_name, best_name)})")

    st.markdown("---")
    st.caption("Usage academique uniquement")

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="main-header">
    <h1 style="margin:0;font-size:1.6rem;">
        Facteurs associes a la Cesarienne au Cameroun
    </h1>
    <p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.9;">
        {page}
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSE DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════
if page == "Analyse descriptive":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total naissances",       f"{n_total:,}")
    c2.metric("Cesariennes",            f"{n_ces:,}",           f"{taux:.1f}%")
    c3.metric("Voie basse",             f"{n_total-n_ces:,}",   f"{100-taux:.1f}%")
    c4.metric("Regions couvertes",      df["region"].nunique())
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(
            labels=["Voie basse","Cesarienne"],
            values=[n_total-n_ces, n_ces],
            hole=0.42, marker_colors=["#2e86ab","#e74c3c"],
            textinfo="label+percent", textfont_size=13
        ))
        fig.update_layout(title="Mode d accouchement", height=360,
            showlegend=False,
            annotations=[dict(text=f"{taux:.1f}%<br>C-section",
                x=0.5, y=0.5, font_size=14, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        reg = df.groupby("region")["cesarienne"].agg(["mean","count"]).reset_index()
        reg.columns = ["region","taux","n"]
        reg["pct"] = (reg["taux"]*100).round(1)
        reg = reg.sort_values("pct")
        fig2 = px.bar(reg, x="pct", y="region", orientation="h",
            text="pct", color="pct", color_continuous_scale="RdYlGn_r",
            title="Taux de cesarienne par region (%)",
            labels={"pct":"Taux (%)","region":"Region"})
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Distribution par variable")
    vars_desc = {
        "age_groupe":"Groupe d age","milieu":"Milieu",
        "instruction":"Instruction","richesse":"Richesse",
        "parite":"Parite","statut_union":"Statut matrimonial",
        "sexe_enfant":"Sexe enfant","cva":"Visites prenatales",
        "lieu_accouchement":"Lieu d accouchement"
    }
    var_c = st.selectbox("Variable", list(vars_desc.keys()),
                          format_func=lambda x: vars_desc[x])
    ct     = pd.crosstab(df[var_c], df["cesarienne"])
    ct_pct = pd.crosstab(df[var_c], df["cesarienne"], normalize="index")*100
    plot_df = pd.DataFrame({
        "Modalite":    ct.index.astype(str),
        "Voie basse":  ct_pct[0].round(1) if 0 in ct_pct.columns else [0]*len(ct),
        "Cesarienne":  ct_pct[1].round(1) if 1 in ct_pct.columns else [0]*len(ct),
    })
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Voie basse", x=plot_df["Modalite"],
                           y=plot_df["Voie basse"], marker_color="#2e86ab"))
    fig3.add_trace(go.Bar(name="Cesarienne", x=plot_df["Modalite"],
                           y=plot_df["Cesarienne"], marker_color="#e74c3c"))
    fig3.update_layout(barmode="stack", height=360,
        title=f"Repartition par {vars_desc[var_c]}", yaxis_title="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE BIVARIEE
# ══════════════════════════════════════════════════════════════════════
elif page == "Analyse bivariee":
    st.subheader("Tests Chi2 — association avec la cesarienne")

    rows = []
    for var, info in biv.items():
        p = info["p_value"]
        for mod, dat in info["data"].items():
            rows.append({
                "Variable":     info["label"],
                "Modalite":     mod,
                "n":            dat["n"],
                "n cesarienne": dat["n_ces"],
                "Taux (%)":     dat["pct_ces"],
                "p Chi2":       p,
                "Sig.":         "Oui ***" if p<0.001 else "Oui **" if p<0.01 else "Oui *" if p<0.05 else "NS"
            })
    df_biv = pd.DataFrame(rows)
    filtre = st.selectbox("Filtrer", ["Toutes","Significatives","Non significatives"])
    if filtre == "Significatives":
        df_biv = df_biv[df_biv["Sig."] != "NS"]
    elif filtre == "Non significatives":
        df_biv = df_biv[df_biv["Sig."] == "NS"]

    st.dataframe(
        df_biv.style
              .background_gradient(subset=["Taux (%)"], cmap="Reds")
              .format({"Taux (%)":"{:.1f}", "p Chi2":"{:.4f}"}),
        use_container_width=True, height=400
    )

    st.subheader("Graphique par variable")
    var_biv = st.selectbox("Variable", list(biv.keys()),
                            format_func=lambda x: biv[x]["label"])
    d = biv[var_biv]
    mods = list(d["data"].keys())
    taux_v = [d["data"][m]["pct_ces"] for m in mods]
    fig4 = px.bar(x=mods, y=taux_v, text=taux_v, color=taux_v,
        color_continuous_scale="Reds",
        title=f"{d['label']} — p = {d['p_value']:.4f}",
        labels={"x":d["label"],"y":"Taux (%)"})
    fig4.add_hline(y=taux, line_dash="dash", line_color="#2e86ab",
                   annotation_text=f"Moy. nationale : {taux:.1f}%")
    fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig4.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — REGRESSION LOGISTIQUE
# ══════════════════════════════════════════════════════════════════════
elif page == "Regression logistique":
    st.subheader("Regression logistique multivariee — Odds Ratios")
    st.caption("Ref : 20-34 ans | Rural | Aucune instruction | Tres pauvre | "
               "Multipare | En union | Feminin | 4 visites et plus")

    df_logit = pd.DataFrame(logit_rows)
    df_logit = df_logit[df_logit["variable"] != "const"].copy()

    c1, c2 = st.columns([3,2])
    with c1:
        df_s = df_logit.sort_values("OR", ascending=True)
        fig5 = go.Figure()
        for _, row in df_s.iterrows():
            color = "#e74c3c" if row["significant"] else "#7f8c8d"
            fig5.add_trace(go.Scatter(
                x=[row["CI_low"], row["OR"], row["CI_high"]],
                y=[row["variable"]]*3,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=[5,11,5], color=color),
                showlegend=False
            ))
        fig5.add_vline(x=1, line_dash="dash", line_color="#2e86ab", line_width=2)
        fig5.update_layout(
            title="Forest Plot — OR ajustes (IC 95%)",
            xaxis_title="Odds Ratio (echelle log)",
            xaxis_type="log", height=480,
            yaxis=dict(tickfont=dict(size=10))
        )
        st.plotly_chart(fig5, use_container_width=True)

    with c2:
        df_d = df_logit[["variable","OR","CI_low","CI_high","p_value","significant"]].copy()
        df_d.columns = ["Variable","OR","IC2.5%","IC97.5%","p","Sig."]
        df_d["Sig."] = df_d["Sig."].map({True:"***" if True else "ns", False:"ns"})
        df_d["Sig."] = df_logit["p_value"].apply(
            lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns")
        st.dataframe(
            df_d.style.format({"OR":"{:.2f}","IC2.5%":"{:.2f}","IC97.5%":"{:.2f}","p":"{:.4f}"}),
            use_container_width=True, height=440
        )

    # Facteurs significatifs — cartes natives Streamlit (pas de HTML)
    sig_df = df_logit[df_logit["significant"]].sort_values("OR", ascending=False)
    st.subheader(f"Facteurs significativement associes ({len(sig_df)} variables, p < 0.05)")

    cols3 = st.columns(3)
    for i, (_, row) in enumerate(sig_df.iterrows()):
        with cols3[i % 3]:
            direction = "Augmente le risque" if row["OR"] > 1 else "Diminue le risque"
            delta_str = f"OR = {row['OR']:.2f}"
            # Utiliser st.metric natif — rendu garanti
            st.metric(
                label=row["variable"],
                value=delta_str,
                delta=direction,
                delta_color="inverse" if row["OR"] < 1 else "normal"
            )
            st.caption(f"IC 95% : [{row['CI_low']:.2f} – {row['CI_high']:.2f}] | p = {row['p_value']:.4f}")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════
elif page == "Machine Learning":
    st.subheader("Comparaison des 6 algorithmes de Machine Learning")

    if not models:
        st.error("Aucun modele trouve. Executez setup.py localement puis "
                 "committez les fichiers .pkl sur GitHub.")
        st.code("python setup.py", language="bash")
        st.stop()

    if not ml_metrics:
        st.warning("Metriques non trouvees. Relancez setup.py.")
        st.stop()

    # ── Tableau comparatif des 6 modeles ──────────────────────────────
    met_rows = []
    for k in ["logistic_regression","gradient_boosting","random_forest",
              "xgboost","decision_tree","knn"]:
        if k in ml_metrics:
            r = ml_metrics[k]
            met_rows.append({
                "Modele":     MODEL_NAMES[k],
                "AUC-ROC":   r["auc"],
                "F1-Score":  r["f1"],
                "Rappel":    r["recall"],
                "Precision": r["precision"],
                "Exactitude":r["accuracy"],
            })
    df_m = pd.DataFrame(met_rows)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Tableau comparatif — 6 modeles**")
        st.dataframe(
            df_m.style
                .highlight_max(
                    subset=["AUC-ROC","F1-Score","Rappel","Precision","Exactitude"],
                    color="#1e3a28"
                )
                .format({c: "{:.3f}" for c in
                         ["AUC-ROC","F1-Score","Rappel","Precision","Exactitude"]}),
            use_container_width=True, height=280
        )
        best_row = df_m.loc[df_m["AUC-ROC"].idxmax()]
        st.success(f"Meilleur modele (AUC) : **{best_row['Modele']}** "
                   f"— AUC = {best_row['AUC-ROC']:.3f}")
        st.info("AUC 0.70–0.80 = acceptable | 0.80–0.90 = bon | > 0.90 = excellent")
        st.caption("Metrique principale : AUC-ROC (insensible au desequilibre des classes)")

    with c2:
        # Graphique barres horizontales AUC — plus lisible que radar pour 6 modeles
        df_m_sorted = df_m.sort_values("AUC-ROC", ascending=True)
        colors_bar = [MODEL_COLORS.get(
            [k for k, v in MODEL_NAMES.items() if v == row["Modele"]][0], "#999"
        ) for _, row in df_m_sorted.iterrows()]
        fig_auc = go.Figure(go.Bar(
            x=df_m_sorted["AUC-ROC"],
            y=df_m_sorted["Modele"],
            orientation="h",
            marker_color=colors_bar,
            text=df_m_sorted["AUC-ROC"].apply(lambda x: f"{x:.3f}"),
            textposition="outside"
        ))
        fig_auc.add_vline(x=0.7, line_dash="dash", line_color="#f39c12",
                          annotation_text="Seuil acceptable (0.70)")
        fig_auc.add_vline(x=0.8, line_dash="dash", line_color="#27ae60",
                          annotation_text="Seuil bon (0.80)")
        fig_auc.update_layout(
            title="AUC-ROC par algorithme",
            xaxis=dict(range=[0.5, 0.87], title="AUC-ROC"),
            height=320, margin=dict(l=0, r=80)
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    # ── Radar chart multi-metriques ────────────────────────────────────
    st.subheader("Profil multi-metriques des 6 modeles")
    cats = ["AUC-ROC", "F1-Score", "Rappel", "Precision", "Exactitude"]
    fig_radar = go.Figure()
    for _, row in df_m.iterrows():
        key = [k for k, v in MODEL_NAMES.items() if v == row["Modele"]][0]
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[c] for c in cats], theta=cats, fill="toself",
            name=row["Modele"], line_color=MODEL_COLORS[key], opacity=0.6))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=420, title="Comparaison multi-metriques — 6 algorithmes"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Importance Gini — Random Forest ───────────────────────────────
    if not feat_imp.empty:
        st.subheader("Importance des variables — Random Forest (score Gini)")
        top_n = st.slider("Nombre de variables a afficher", 10,
                           min(len(feat_imp), 40), 20)
        fi_t = feat_imp.head(top_n).sort_values("Importance")
        fig7 = px.bar(fi_t, x="Importance", y="Variable", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            title=f"Top {top_n} variables — Importance Gini (Random Forest)",
            labels={"Importance": "Score Gini moyen", "Variable": ""})
        fig7.update_layout(height=max(380, top_n * 22),
                           coloraxis_showscale=False)
        st.plotly_chart(fig7, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — ANALYSE SHAP
# ══════════════════════════════════════════════════════════════════════
elif page == "Analyse SHAP":
    st.subheader("Analyse SHAP — Explicabilite du modele Random Forest")
    st.caption(
        "SHAP (SHapley Additive exPlanations) quantifie la contribution de "
        "chaque variable a la prediction individuelle du modele."
    )

    if shap_data is None:
        st.error("Fichier shap_results.pkl introuvable. Executez setup.py.")
        st.code("python setup.py", language="bash")
        st.stop()

    shap_vals_list   = shap_data["shap_values"]     # liste (n, p)
    feature_names    = shap_data["feature_names"]
    shap_imp_records = shap_data["shap_importance"] # liste de dicts
    X_shap_list      = shap_data["X_shap"]
    n_samples        = shap_data["n_samples"]

    shap_arr  = np.array(shap_vals_list)   # (n, p)
    X_shap_np = np.array(X_shap_list)      # (n, p)
    shap_imp_df = pd.DataFrame(shap_imp_records).sort_values(
        "SHAP_importance", ascending=False).reset_index(drop=True)

    st.info(f"Analyse realisee sur {n_samples} observations de l ensemble de test "
            f"| Modele : Random Forest (meilleur AUC stable parmi les modeles tree-based)")

    # ── 1. Barplot importance SHAP ────────────────────────────────────
    st.subheader("1. Importance globale SHAP (valeur absolue moyenne)")
    st.markdown(
        "Cette mesure indique quelles variables influencent **le plus** "
        "les predictions du modele, toutes observations confondues."
    )
    top_s = st.slider("Nombre de variables SHAP", 10,
                       min(len(shap_imp_df), 34), 20, key="shap_slider")
    shap_top = shap_imp_df.head(top_s).sort_values("SHAP_importance")
    fig_shap_bar = px.bar(
        shap_top, x="SHAP_importance", y="Variable", orientation="h",
        color="SHAP_importance", color_continuous_scale="Oranges",
        title=f"Top {top_s} — Importance SHAP moyenne |SHAP|",
        labels={"SHAP_importance": "Importance SHAP (|valeur| moyenne)", "Variable": ""}
    )
    fig_shap_bar.update_layout(height=max(380, top_s * 22),
                                coloraxis_showscale=False)
    st.plotly_chart(fig_shap_bar, use_container_width=True)

    # ── 2. Comparaison SHAP vs Gini ───────────────────────────────────
    st.subheader("2. Comparaison SHAP vs importance Gini")
    st.markdown(
        "SHAP est plus fiable que le score Gini pour les variables categorielles "
        "et les variables avec biais de cardinalite."
    )
    if not feat_imp.empty:
        # Aligner les deux rankings sur les 15 premieres variables SHAP
        top15_shap = shap_imp_df.head(15)["Variable"].tolist()
        gini_vals  = feat_imp.set_index("Variable")["Importance"].reindex(
            top15_shap).fillna(0)
        shap_vals_top = shap_imp_df.set_index("Variable")[
            "SHAP_importance"].reindex(top15_shap).fillna(0)

        # Normaliser pour comparaison visuelle
        gini_norm = (gini_vals / gini_vals.max()).values
        shap_norm = (shap_vals_top / shap_vals_top.max()).values

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="SHAP (normalise)", x=top15_shap,
            y=shap_norm, marker_color="#e67e22", opacity=0.85))
        fig_comp.add_trace(go.Bar(
            name="Gini (normalise)", x=top15_shap,
            y=gini_norm, marker_color="#2e86ab", opacity=0.85))
        fig_comp.update_layout(
            barmode="group", height=400,
            title="Importance SHAP vs Gini — Top 15 variables (normalisees)",
            xaxis=dict(tickangle=-35),
            yaxis_title="Importance normalisee [0-1]",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── 3. Beeswarm SHAP (dot plot) ───────────────────────────────────
    st.subheader("3. Distribution des valeurs SHAP par variable (dot plot)")
    st.markdown(
        "Chaque point = une observation. **Rouge** = valeur de feature elevee, "
        "**Bleu** = valeur de feature faible. La position sur l axe X indique "
        "si la variable **augmente** (droite) ou **diminue** (gauche) la probabilite de cesarienne."
    )
    n_vars_bee = st.slider("Nombre de variables", 5, min(20, len(shap_imp_df)), 12,
                            key="bee_slider")
    top_vars   = shap_imp_df.head(n_vars_bee)["Variable"].tolist()

    fig_bee = go.Figure()
    for i, var in enumerate(reversed(top_vars)):
        try:
            col_idx = feature_names.index(var)
        except ValueError:
            continue
        shap_col  = shap_arr[:, col_idx]
        feat_col  = X_shap_np[:, col_idx]
        # Normaliser les valeurs pour la couleur
        feat_norm = (feat_col - feat_col.min()) / (feat_col.max() - feat_col.min() + 1e-8)
        # Jitter vertical
        jitter = np.random.uniform(-0.3, 0.3, len(shap_col))
        fig_bee.add_trace(go.Scatter(
            x=shap_col,
            y=np.full(len(shap_col), i) + jitter,
            mode="markers",
            marker=dict(
                size=4, opacity=0.6,
                color=feat_norm,
                colorscale="RdBu_r",
                showscale=(i == 0),
                colorbar=dict(
                    title="Valeur<br>feature",
                    tickvals=[0, 1],
                    ticktext=["Faible", "Elevee"],
                    len=0.5, y=0.5
                ) if i == 0 else None
            ),
            name=var, showlegend=False,
            hovertemplate=f"<b>{var}</b><br>SHAP: %{{x:.4f}}<extra></extra>"
        ))

    fig_bee.add_vline(x=0, line_color="white", line_width=1.5, line_dash="dash")
    fig_bee.update_layout(
        height=max(450, n_vars_bee * 40),
        title="Beeswarm plot — Distribution des valeurs SHAP",
        xaxis_title="Valeur SHAP (impact sur la prediction)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_vars_bee)),
            ticktext=list(reversed(top_vars)),
            tickfont=dict(size=10)
        ),
        margin=dict(l=160)
    )
    st.plotly_chart(fig_bee, use_container_width=True)

    # ── 4. Interpretation narrative ───────────────────────────────────
    st.subheader("4. Interpretation des resultats SHAP")
    top3 = shap_imp_df.head(3)["Variable"].tolist()

    cols_interp = st.columns(3)
    for i, var in enumerate(top3):
        col_idx   = feature_names.index(var) if var in feature_names else -1
        shap_mean = float(shap_imp_df.loc[shap_imp_df["Variable"]==var,
                                          "SHAP_importance"].values[0])
        with cols_interp[i]:
            st.metric(
                label=f"#{i+1} Variable la plus importante",
                value=var,
                delta=f"SHAP moy. = {shap_mean:.5f}"
            )
            if col_idx >= 0:
                shap_col  = shap_arr[:, col_idx]
                pos_pct   = (shap_col > 0).mean() * 100
                st.caption(
                    f"Dans {pos_pct:.0f}% des cas, cette variable "
                    f"**augmente** la probabilite de cesarienne."
                )

    st.markdown("---")
    st.markdown(
        "**Lecture des valeurs SHAP :** Une valeur SHAP positive pour une observation "
        "signifie que cette variable a contribue a **augmenter** la probabilite predite "
        "de cesarienne par rapport a la prediction de base du modele. Une valeur negative "
        "signifie qu elle a contribue a la **diminuer**. La magnitude indique l importance "
        "de cette contribution."
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICTION
# ══════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.subheader("Prediction du risque de cesarienne — profil individuel")
    st.info("Outil pedagogique et de recherche uniquement — pas un outil clinique.")

    if not models:
        st.error("Modeles ML non charges. Voir onglet Machine Learning.")
        st.stop()

    c_form, c_res = st.columns([1,1])
    with c_form:
        st.markdown("**Caracteristiques de la femme**")
        age_i  = st.selectbox("Groupe d age", ["Moins_20","20_34","35_plus"],
                               format_func=lambda x:{"Moins_20":"< 20 ans","20_34":"20-34 ans","35_plus":">= 35 ans"}[x])
        mil_i  = st.selectbox("Milieu",   ["Urbain","Rural"])
        ins_i  = st.selectbox("Instruction", ["Aucune","Primaire","Secondaire","Superieur"])
        ric_i  = st.selectbox("Richesse",
            ["Tres_pauvre","Pauvre","Moyen","Riche","Tres_riche"],
            format_func=lambda x: x.replace("_"," "))
        par_i  = st.selectbox("Parite",   ["Primipare","Multipare"])
        stat_i = st.selectbox("Statut",   ["En_union","Pas_en_union"],
                               format_func=lambda x: x.replace("_"," "))
        sex_i  = st.selectbox("Sexe enfant", ["Masculin","Feminin"])
        cva_i  = st.selectbox("Visites prenatales", ["4_plus","Moins_4"],
                               format_func=lambda x:{"4_plus":"4 et plus","Moins_4":"Moins de 4"}[x])
        reg_i  = st.selectbox("Region", sorted(df["region"].unique()))
        mod_c  = st.selectbox("Modele ML a utiliser",
                               list(models.keys()),
                               format_func=lambda x: MODEL_NAMES.get(x, x))
        btn = st.button("Calculer le risque", type="primary", use_container_width=True)

    with c_res:
        if btn:
            inp = {c: 0 for c in feat_cols}
            for k in [f"age_groupe_{age_i}", f"milieu_{mil_i}",
                       f"instruction_{ins_i}", f"richesse_{ric_i}",
                       f"parite_{par_i}", f"statut_union_{stat_i}",
                       f"sexe_enfant_{sex_i}", f"cva_{cva_i}",
                       f"region_{reg_i}"]:
                if k in inp:
                    inp[k] = 1

            X_pred = pd.DataFrame([inp])[feat_cols]
            # Appliquer le scaler pour les modeles qui en ont besoin
            if mod_c in SCALED_MODELS and scaler is not None:
                X_pred_arr = scaler.transform(X_pred)
                proba = models[mod_c].predict_proba(X_pred_arr)[0][1]
            else:
                proba = models[mod_c].predict_proba(X_pred)[0][1]
            pct = proba * 100

            if   pct >= 20: css,icon,niv = "pred-high","🔴","RISQUE ELEVE"
            elif pct >= 10: css,icon,niv = "pred-mod", "🟡","RISQUE MODERE"
            else:           css,icon,niv = "pred-low", "🟢","RISQUE FAIBLE"

            st.markdown(f"""
            <div class="pred-box {css}">
                {icon} &nbsp; {niv}<br>
                <span style="font-size:2.8rem;">{pct:.1f}%</span><br>
                <span style="font-size:0.85rem;font-weight:normal;">
                    probabilite estimee de cesarienne<br>
                    Modele : {MODEL_NAMES[mod_c]}
                </span>
            </div>
            """, unsafe_allow_html=True)

            fig8 = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={"suffix":"%","font":{"size":28}},
                gauge={
                    "axis":{"range":[0,50],"ticksuffix":"%"},
                    "bar":{"color":"#e74c3c" if pct>=20 else "#f39c12" if pct>=10 else "#27ae60"},
                    "steps":[{"range":[0,10],"color":"#1a3a2a"},
                              {"range":[10,20],"color":"#3a3010"},
                              {"range":[20,50],"color":"#3a1a1a"}],
                    "threshold":{"line":{"color":"#2e86ab","width":3},
                                 "thickness":0.75,"value":taux}
                },
                title={"text":"Probabilite estimee"}
            ))
            fig8.update_layout(height=260)
            st.plotly_chart(fig8, use_container_width=True)

            ratio = pct/taux
            st.metric("Par rapport a la moyenne nationale",
                      f"{pct:.1f}%",
                      delta=f"x{ratio:.1f} la moyenne ({taux:.1f}%)",
                      delta_color="inverse" if ratio < 1 else "normal")
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;opacity:0.5;">
                <div style="font-size:3.5rem;">🔮</div>
                <p>Remplissez le formulaire et cliquez<br>
                sur <b>Calculer le risque</b></p>
            </div>
            """, unsafe_allow_html=True)
