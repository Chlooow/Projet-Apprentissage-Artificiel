import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# fonction auxiliaires
# def regroup_category2(category: str) -> str:
#     """
#     Regroupe les catégories de niveau 2 selon une logique métier tout en
#     préservant les catégories à forte valeur predictive.
#     """
#     # category = category.lower()
#     # Regroupement des articles pour enfants
#     if category in ['BABY BOYS', 'BABY GIRLS', 'NEWBORN', 'NEWBORN GIFT SETS', 'GIRLS', 'BOYS']:
#         return 'CHILDREN'
    
#     # Regroupement des articles de "haut luxe"
#     elif category in ['EXCEPTIONAL TIMEPIECES', 'TIMEPIECES']:
#         return 'TIMEPIECES_GENERAL'
#     elif category in ['JEWELS', 'JEWELLERY']:
#         return 'JEWELLERY_GENERAL'
#     # # Regroupement des maroquineries et accessoires en cuir
#     # elif category in ['SMALL LEATHER GOODS', 'LEATHER GOODS']:
#     #     return 'LEATHER_GOODS_GENERAL'
    
#     elif category in ['HANDBAGS', 'SHOES', 'CLOTHING', 'MAISON', 'ACCESSORIES']:
#         return category
        
#     # Pour toute autre catégorie non identifiée
#     else:
#         return 'OTHER_C2'
   
def preprocess_dior(df):
    """ 
    cette fonction permet de nettoyer notre dataset actuelle
    """
    df = df.copy()

    # les colonnes à supprimer
    cols_to_drop = [
        'website_name', 'competence_date', 'country_code', 'currency_code',
        'brand', 'flg_discount', 'full_price', 'full_price_eur','price_eur',
        'product_code', 'itemurl', 'imageurl', 'title'
    ]
    df = df.drop(columns=cols_to_drop)

    # Supprimer la ligne avec une valeur manquante dans category3_code
    df['category3_code'] = df['category3_code'].replace('N.A.', np.nan)
    df = df.dropna(subset=['category3_code'])
    
    # Separer les targets
    features = df.drop(columns=['price'])
    targets = df[['price']]

    return features, targets