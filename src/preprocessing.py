import pandas as pd
from pypinyin import lazy_pinyin

# fonction auxiliaires
def romanize_title(text: str) -> str:
    """
    Transforme un texte contenant des caractères chinois
    en romanisation (pinyin).
    """
    if not isinstance(text, str):
        return ""
    return " ".join(lazy_pinyin(text))

def preprocess_dior(df):
    """ 
    cette fonction permet de nettoyer notre dataset actuelle
    Returns:
        df_features: DataFrame des features prêtes pour le modèle
        df_targets: DataFrame avec les colonnes 'price' et 'price_eur'
    """

    # les colonnes à supprimer
    cols_to_drop = [
        'website_name', 'competence_date', 'country_code', 'currency_code',
        'brand', 'flg_discount', 'full_price', 'full_price_eur',
        'product_code', 'itemurl', 'imageurl'
    ]
    df = df.drop(columns=cols_to_drop)

    # Supprimer la ligne avec une valeur manquante dans category3_code
    df = df.dropna(subset=['category3_code'])

    # Romanisation des titres
    df['title_romanized'] = df['title'].apply(romanize_title)

    # Separer les targets
    targets = df[['price', 'price_eur']]
    features = df.drop(columns=['price', 'price_eur'])

    return features, targets





