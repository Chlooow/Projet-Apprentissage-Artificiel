import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import os
import sys
sys.path.append(os.path.abspath("./src"))
from src.modelisation import count_encoding
from pypinyin import lazy_pinyin
from sklearn.preprocessing import OneHotEncoder
# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="Dior China Price Prediction",
    layout="wide"
)

def load_dior():
    dataset = load_dataset(
        "DBQ/Dior.Product.prices.China",
        split="train"
    )
    df = dataset.to_pandas()
    return df

df = load_dior()

# -----------------------
# LOAD ARTIFACTS
# -----------------------
model_GB = joblib.load("artifacts/model_gb.pkl")
counts_cat1_GB = joblib.load("artifacts/count_cat1.pkl")
counts_cat2_GB = joblib.load("artifacts/count_cat2.pkl")
counts_cat3_GB = joblib.load("artifacts/count_cat3.pkl")

model_RF = joblib.load("artifacts/model_rf.pkl")
counts_cat1_RF = joblib.load("artifacts/count_rf_cat1.pkl")
counts_cat2_RF = joblib.load("artifacts/count_rf_cat2.pkl")
counts_cat3_RF = joblib.load("artifacts/count_rf_cat3.pkl")

transform_info = joblib.load("artifacts/target_transform.pkl")

# -----------------------
# TITLE
# -----------------------
st.title("Dior China - Price Prediction")

# ======================================================
# SECTION 1 — PRESENTATION
# ======================================================
st.header("Présentation du projet")

st.markdown("""
## Techniques D'apprentissage Artificielles - Projet Final 2025-2026
#### Prédiction de prix de Dior Chine
Dans le cadre du cours Apprentissage Artificiel 2025-2026 avec Madame Rakia Jaziri
##### Informations 
- **Chef de projet etudiant**: Chloé Nsonde Makoundou 
- **Manager** : Madame Rakia Jaziri 
- **Formation** : Master 1 IBD  
- **type de projet**: individuel

##### Problématique
*Comment modéliser et prédire efficacement les prix des produits Dior en Chine, et quels sont les facteurs déterminants qui influencent le pricing dans le secteur du luxe ?*

##### Dataset
lien du dataset : https://huggingface.co/datasets/DBQ/Dior.Product.prices.China
""")

# -----------------------
# Choix du modèle
# -----------------------
selected_model_name = st.selectbox(
    "Choisissez le modèle",
    ["Random Forest", "Gradient Boost"]
)

if selected_model_name == "Random Forest":
    model = model_RF
    counts_cat1 = counts_cat1_RF
    counts_cat2 = counts_cat2_RF
    counts_cat3 = counts_cat3_RF
else:
    model = model_GB
    counts_cat1 = counts_cat1_GB
    counts_cat2 = counts_cat2_GB
    counts_cat3 = counts_cat3_GB


# ======================================================
# SECTION 2 — PREDICTION
# ======================================================

st.header("Prédire le prix d’un produit")

col1, col2, col3 = st.columns(3)

with col1:
    category1 = st.selectbox("Catégorie principale", df["category1_code"].unique())

with col2:
    category2 = st.selectbox("Catégorie secondaire", df["category2_code"].unique())

with col3:
    category3 = st.selectbox("Catégorie fine", df["category3_code"].unique())

# Bouton prédiction
if st.button("Prédire le prix"):
    # Utilisation de l'encodage correspondant au modèle
    def get_count(counts, cat):
        if isinstance(counts, (dict, pd.Series)):
            return counts.get(cat, 0)
        elif isinstance(counts, OneHotEncoder):
            # transforme en 2D array
            X_transformed = counts.transform([[cat]])
            # Si sparse, convertir en array
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            # On retourne la ligne 0
            return X_transformed[0]
        else:
            raise ValueError(f"Type d'encodeur inconnu : {type(counts)}")

    cat1_count = get_count(counts_cat1, category1)
    cat2_count = get_count(counts_cat2, category2)
    cat3_count = get_count(counts_cat3, category3)

    X_input = np.hstack([cat1_count, cat2_count, cat3_count]).reshape(1, -1)
    log_price = model.predict(X_input)[0]
    price = np.expm1(log_price)


    st.success(f"Prix estimé : **{price:,.0f} CNY**")

    sql_query = f"""
    SELECT title, price
    FROM dior
    WHERE category1_code = '{category1}'
    AND category2_code = '{category2}'
    AND category3_code = '{category3}';
    """
    st.markdown("### Requête SQL associée")
    st.code(sql_query, language="sql")

filtered_df = df[
    (df["category1_code"] == category1) &
    (df["category2_code"] == category2) &
    (df["category3_code"] == category3)
][["title", "price"]]
st.markdown("### 📊 Produits Dior correspondant à ces catégories")
st.dataframe(filtered_df, use_container_width=True)

st.markdown("### 📈 Statistiques sur les prix réels")

st.write({
    "Nombre de produits": len(filtered_df),
    "Prix minimum": filtered_df["price"].min(),
    "Prix moyen": round(filtered_df["price"].mean(), 2),
    "Prix maximum": filtered_df["price"].max()
})
