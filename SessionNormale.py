# =============================================================================
# APPLICATION STREAMLIT - ANALYSE DES MALADIES DES PLANTES
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(
    page_title="Maladies des Plantes",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# STYLES 


st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Georgia', serif; }

.stApp { background-color: #f5f7f0; }

[data-testid="stSidebar"] { background-color: #1c2e1c; }
[data-testid="stSidebar"] * { color: #d4e6c3 !important; }

h1 {
    color: #1c2e1c; font-size: 2rem; font-weight: 700;
    border-bottom: 3px solid #4a7c4e;
    padding-bottom: 0.4rem; margin-bottom: 1.5rem;
}
h2 { color: #2e4a2e; font-size: 1.3rem; margin-top: 1.5rem; }
h3 { color: #3a5c3a; }

[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #c5d9b5;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
[data-testid="metric-container"] label {
    color: #6a8a5a !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1c2e1c; font-size: 1.8rem; font-weight: 700;
}

.stButton > button {
    background-color: #2e6b30; color: white;
    border: none; border-radius: 8px;
    padding: 0.55rem 1.6rem;
    font-size: 0.95rem; font-weight: 600;
    transition: background 0.2s;
}
.stButton > button:hover { background-color: #1c4a1e; color: white; }

[data-testid="stDataFrame"] {
    border: 1px solid #c5d9b5; border-radius: 8px; overflow: hidden;
}

.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1200px; }
hr { border-color: #c5d9b5; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2e6b30","#6aaa6e","#a8d5a2","#c8e6c9","#f4a261","#e76f51","#264653","#2a9d8f"]


# CHARGEMENT DU DATASET


@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# PRÉTRAITEMENT DES DATA


@st.cache_data
def preprocess_data(df):
    data = df.copy()
    data = data.drop_duplicates(keep='first')

    for col in ['leaf_length', 'leaf_width', 'stem_diameter']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].median())

    for col in ['soil_type', 'weather', 'pesticide', 'disease_type']:
        data[col] = data[col].astype(str).fillna(data[col].mode()[0])

    data['leaf_ratio'] = (data['leaf_length'] / data['leaf_width'].replace(0, np.nan)).fillna(0)
    data['leaf_area']  = data['leaf_length'] * data['leaf_width']

    data['pesticide_encoded'] = data['pesticide'].str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    soil_types    = sorted(data['soil_type'].unique())
    weather_types = sorted(data['weather'].unique())
    for s in soil_types:
        data[f'soil_{s}'] = (data['soil_type'] == s).astype(int)
    for w in weather_types:
        data[f'weather_{w}'] = (data['weather'] == w).astype(int)

    numeric_cols = ['leaf_length', 'leaf_width', 'stem_diameter', 'leaf_ratio', 'leaf_area']
    scaler = StandardScaler()
    for col in numeric_cols:
        data[f'{col}_scaled'] = scaler.fit_transform(data[[col]])

    le = LabelEncoder()
    data['disease_encoded'] = le.fit_transform(data['disease_type'])

    feature_columns = (
        ['leaf_length_scaled','leaf_width_scaled','stem_diameter_scaled',
         'leaf_ratio_scaled','leaf_area_scaled','pesticide_encoded']
        + [f'soil_{s}' for s in soil_types]
        + [f'weather_{w}' for w in weather_types]
    )

    X = data[feature_columns].astype(float)
    y = data['disease_encoded'].astype(int)

    disease_map = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    inv_map     = {v: k for k, v in disease_map.items()}

    stats = {
        'soil_types':      soil_types,
        'weather_types':   weather_types,
        'feature_columns': feature_columns,
        'means': {col: float(data[col].mean()) for col in numeric_cols},
        'stds':  {col: float(data[col].std())  for col in numeric_cols},
    }

    return data, X, y, disease_map, inv_map, stats, le


# SIDEBAR


st.sidebar.markdown("## Maladies des Plantes")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV", type=['csv'])
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", ["Accueil","Exploration","Prétraitement","Visualisation","Modélisation"])


# ACCUEIL


if menu == "Accueil":
    st.title("Analyse des maladies des plantes")
    st.markdown("""
Cette application permet d'explorer, de visualiser et de classifier les maladies de plantes
à partir de mesures morphologiques et de données environnementales.
    """)
    c1, c2 = st.columns(2)
    #Ecrivons en markdown pour une bonne przsentation des variables
    with c1:
        st.markdown("""
**Variables du jeu de données**

| Variable | Description |
|---|---|
| `leaf_length` | Longueur de la feuille  |
| `leaf_width` | Largeur de la feuille    |
| `stem_diameter` | Diamètre de la tige   |
| `soil_type` | Type de sol |
| `weather` | Conditions météorologiques |
| `pesticide` | Utilisation de pesticide (yes/no) |
| `disease_type` | Maladie observée — **cible** |
        """)
    with c2:
        st.markdown("""
**Fonctionnalités disponibles**

1. **Exploration** — statistiques descriptives et aperçu du jeu de données
2. **Prétraitement** — nettoyage, encodage et normalisation
3. **Visualisation** — graphiques interactifs par variable et par maladie
4. **Modélisation** — entraînement d'un Random Forest, évaluation et prédiction
        """)
    if uploaded_file:
        st.success("Fichier chargé. Utilisez le menu latéral pour naviguer.")
    else:
        st.info("Chargez un fichier CSV depuis le panneau latéral pour commencer.")

# EXPLORATION

elif menu == "Exploration":
    st.title("Exploration du jeu de données")
    if not uploaded_file:
        st.warning("Aucun fichier chargé.")
        st.stop()

    df = load_data(uploaded_file)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations",      df.shape[0])
    c2.metric("Variables",          df.shape[1])
    c3.metric("Doublons",           int(df.duplicated().sum()))
    c4.metric("Valeurs manquantes", int(df.isnull().sum().sum()))

    st.markdown("---")
    st.subheader("Aperçu des données")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe().round(4), use_container_width=True)

    st.subheader("Valeurs manquantes par colonne")
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Variable','Valeurs manquantes']
    missing = missing[missing['Valeurs manquantes'] > 0]
    if missing.empty:
        st.success("Aucune valeur manquante détectée.")
    else:
        st.dataframe(missing, use_container_width=True)


# PRÉTRAITEMENT


elif menu == "Prétraitement":
    st.title("Prétraitement des données")
    if not uploaded_file:
        st.warning("Aucun fichier chargé.")
        st.stop()

    df_raw = load_data(uploaded_file)
    data, X, y, disease_map, inv_map, stats, le = preprocess_data(df_raw)

    st.subheader("Étapes appliquées")
    st.markdown("""
1. **Suppression des doublons**
2. **Imputation** : médiane pour les variables numériques, mode pour les catégorielles
3. **Feature engineering** : `leaf_ratio = leaf_length / leaf_width` et `leaf_area = leaf_length * leaf_width`
4. **Encodage de `pesticide`** : yes  1, no  0
5. **One-Hot Encoding** de `soil_type` et `weather`
6. **Normalisation** des variables continues via StandardScaler
7. **Label Encoding** de la variable cible `disease_type`
    """)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", X.shape[0])
    c2.metric("Features",     X.shape[1])
    c3.metric("Classes",      len(disease_map))

    st.subheader("Correspondance des classes")
    st.dataframe(
        pd.DataFrame(list(disease_map.items()), columns=['Maladie','Code numérique']),
        use_container_width=True
    )
    st.subheader("Aperçu du jeu de features normalisé")
    st.dataframe(X.head(8).round(4), use_container_width=True)


# VISUALISATION


elif menu == "Visualisation":
    st.title("Visualisation")
    if not uploaded_file:
        st.warning("Aucun fichier chargé.")
        st.stop()

    df_raw = load_data(uploaded_file)
    data, X, y, disease_map, inv_map, stats, le = preprocess_data(df_raw)

    viz = st.selectbox("Type de graphique", [
        "Répartition des maladies",
        "Distribution des variables",
        "Boîtes à moustaches",
        "Matrice de corrélation",
        "Nuage de points",
    ])

    def styled(fig):
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          font_family='Georgia')
        return fig

    if viz == "Répartition des maladies":
        counts = data['disease_type'].value_counts().reset_index()
        counts.columns = ['Maladie','Effectif']
        c1, c2 = st.columns(2)
        with c1:
            fig = styled(px.bar(counts, x='Maladie', y='Effectif', color='Maladie',
                         title="Effectifs par maladie", color_discrete_sequence=PALETTE))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = styled(px.pie(counts, names='Maladie', values='Effectif',
                         title="Proportions par maladie", color_discrete_sequence=PALETTE))
            st.plotly_chart(fig, use_container_width=True)

    elif viz == "Distribution des variables":
        col = st.selectbox("Variable", ['leaf_length','leaf_width','stem_diameter'])
        fig = styled(px.histogram(data, x=col, color='disease_type', nbins=30,
                     barmode='overlay', opacity=0.75,
                     labels={col: col.replace('_',' ').title(), 'disease_type':'Maladie'},
                     title=f"Distribution de {col} par maladie",
                     color_discrete_sequence=PALETTE))
        st.plotly_chart(fig, use_container_width=True)

    elif viz == "Boîtes à moustaches":
        col = st.selectbox("Variable", ['leaf_length','leaf_width','stem_diameter'])
        fig = styled(px.box(data, x='disease_type', y=col, color='disease_type',
                     labels={'disease_type':'Maladie', col: col.replace('_',' ').title()},
                     title=f"{col.replace('_',' ').title()} selon la maladie",
                     color_discrete_sequence=PALETTE))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif viz == "Matrice de corrélation":
        corr_cols = ['leaf_length','leaf_width','stem_diameter',
                     'leaf_ratio','leaf_area','pesticide_encoded']
        fig = styled(px.imshow(data[corr_cols].corr(), text_auto='.2f', aspect='auto',
                     title="Matrice de corrélation",
                     color_continuous_scale='RdYlGn', zmin=-1, zmax=1))
        st.plotly_chart(fig, use_container_width=True)

    elif viz == "Nuage de points":
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("Axe X", ['leaf_length','leaf_width','stem_diameter'])
        y_col = c2.selectbox("Axe Y", ['leaf_width','leaf_length','stem_diameter','leaf_area'])
        color = st.selectbox("Colorier par", ['disease_type','soil_type','weather'])
        fig = styled(px.scatter(data, x=x_col, y=y_col, color=color,
                     size='stem_diameter', opacity=0.7,
                     title=f"{x_col.replace('_',' ')} vs {y_col.replace('_',' ')}",
                     color_discrete_sequence=PALETTE))
        st.plotly_chart(fig, use_container_width=True)


# MODÉLISATION


elif menu == "Modélisation":
    st.title("Modélisation — Random Forest")
    if not uploaded_file:
        st.warning("Aucun fichier chargé.")
        st.stop()

    df_raw = load_data(uploaded_file)
    data, X, y, disease_map, inv_map, stats, le = preprocess_data(df_raw)

    #  Paramètres du modele
    st.subheader("Paramètres du modèle")
    c1, c2 = st.columns(2)
    test_size = c1.slider("Proportion du jeu de test (%)", 10, 40, 20) / 100
    n_trees   = c2.slider("Nombre d'arbres", 10, 300, 100, step=10)

    if st.button("Lancer l'entraînement"):
        with st.spinner("Entraînement en cours..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            model = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Sauvegarde dans session_state  por que les résultats restent visibles après rechargement
        st.session_state.update({
            'model':   model,
            'labels':  [inv_map[i] for i in sorted(inv_map.keys())],
            'inv_map': inv_map,
            'stats':   stats,
            'acc':     accuracy_score(y_test, y_pred),
            'y_test':  y_test,
            'y_pred':  y_pred,
            'trained': True,
        })

    #  Résultats 
    if st.session_state.get('trained'):
        model   = st.session_state['model']
        labels  = st.session_state['labels']
        inv_map = st.session_state['inv_map']
        stats   = st.session_state['stats']
        acc     = st.session_state['acc']
        y_test  = st.session_state['y_test']
        y_pred  = st.session_state['y_pred']

        st.success(f"Précision sur le jeu de test : **{acc:.2%}**")
        st.markdown("---")

        def styled(fig):
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_family='Georgia')
            return fig

        # Matrice de confusion
        st.subheader("Matrice de confusion")
        cm  = confusion_matrix(y_test, y_pred)
        fig = styled(px.imshow(cm, text_auto=True, x=labels, y=labels,
                     labels=dict(x="Prédit", y="Réel"), color_continuous_scale='Greens'))
        st.plotly_chart(fig, use_container_width=True)

        # Rapport de la classification
        st.subheader("Rapport de classification")
        report = pd.DataFrame(
            classification_report(y_test, y_pred, target_names=labels, output_dict=True)
        ).transpose()
        st.dataframe(report.round(3), use_container_width=True)

        # Importance de mes variables
        st.subheader("Importance des variables")
        imp = pd.DataFrame({'Variable': stats['feature_columns'],
                            'Importance': model.feature_importances_}).sort_values('Importance')
        fig = styled(px.bar(imp, x='Importance', y='Variable', orientation='h',
                     title="Importance des variables (Mean Decrease in Impurity)",
                     color='Importance',
                     color_continuous_scale=[[0,'#c8e6c9'],[1,'#1c4a1e']]))
        fig.update_layout(yaxis_title=None, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Prédiction nouvelle plante
        st.markdown("---")
        st.subheader("Prédiction sur une nouvelle observation")
        st.markdown("Renseignez les caractéristiques de la plante à classifier, puis cliquez sur **Prédire**.")

        c1, c2, c3 = st.columns(3)
        with c1:
            ll   = st.number_input("Longueur de la feuille ", 0.0, 20.0, 10.0, step=0.1, key="ll")
            lw   = st.number_input("Largeur de la feuille cm",  0.0, 10.0,  5.0, step=0.1, key="lw")
        with c2:
            sd   = st.number_input("Diamètre de la tige cm",    0.0,  2.0,  1.0, step=0.05, key="sd")
            pest = st.selectbox("Pesticide utilisé", ['no','yes'], key="pest")
        with c3:
            soil    = st.selectbox("Type de sol",                stats['soil_types'],    key="soil")
            weather = st.selectbox("Conditions météorologiques", stats['weather_types'], key="weather")

        # Bouton hors form — résultat stocké en session_state ne disparaît pas
        if st.button("Prédire la maladie"):
            if lw == 0:
                st.session_state['pred_error'] = "La largeur de la feuille ne peut pas être nulle."
                st.session_state['pred_result'] = None
            else:
                ratio = ll / lw
                area  = ll * lw
                new_data = pd.DataFrame(0.0, index=[0], columns=stats['feature_columns'])
                new_data['pesticide_encoded'] = 1 if pest == 'yes' else 0
                for col_name, value in [('leaf_length',ll),('leaf_width',lw),
                                         ('stem_diameter',sd),('leaf_ratio',ratio),('leaf_area',area)]:
                    std = stats['stds'][col_name]
                    new_data[f'{col_name}_scaled'] = 0.0 if std == 0 else (value - stats['means'][col_name]) / std
                if f'soil_{soil}' in new_data.columns:
                    new_data[f'soil_{soil}'] = 1
                if f'weather_{weather}' in new_data.columns:
                    new_data[f'weather_{weather}'] = 1
                new_data = new_data.astype(float)
                pred   = model.predict(new_data)[0]
                proba  = model.predict_proba(new_data)[0]
                st.session_state['pred_result'] = {
                    'maladie': inv_map[pred],
                    'conf':    proba[pred],
                    'proba':   proba,
                    'labels':  labels,
                }
                st.session_state['pred_error'] = None

        # Affichage du résultat de prediction
        if st.session_state.get('pred_error'):
            st.error(st.session_state['pred_error'])
        elif st.session_state.get('pred_result'):
            res = st.session_state['pred_result']
            st.success(f"Maladie prédite : **{res['maladie'].upper()}** — confiance : {res['conf']:.1%}")
            proba_df = pd.DataFrame({'Maladie': res['labels'], 'Probabilité': res['proba']}).sort_values('Probabilité', ascending=False)
            fig = styled(px.bar(proba_df, x='Maladie', y='Probabilité', color='Maladie',
                         title="Probabilités par classe", color_discrete_sequence=PALETTE,
                         text=proba_df['Probabilité'].map(lambda v: f"{v:.1%}")))
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Lancez l'entraînement pour afficher les résultats et accéder à la prédiction.")