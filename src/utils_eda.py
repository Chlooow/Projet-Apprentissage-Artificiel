import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def check_unique_values(df):
    """
    Affiche le nombre de valeurs uniques par colonne dans un DataFrame.
    """
    print("Nombre de valeurs uniques par colonne:\n")
    for col in df.columns:
        print(f"{col:20s} -> {df[col].nunique()} valeurs uniques")

# Distribution des prix 
def plot_price_distribution(df, column="price"):
    plt.figure(figsize=(8,5))
    sns.histplot(df[column], bins=50, kde=True)
    plt.title(f"Distribution de {column}")
    plt.xlabel(column)
    plt.ylabel("Fréquence")
    plt.show()

    plt.figure(figsize=(8,2))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot de {column}")
    plt.show()


# Nombre de produits par catégorie
def plot_category_counts(df, column):
    plt.figure(figsize=(10,5))
    df[column].value_counts().plot(kind='bar')
    plt.title(f"Nombre de produits par {column}")
    plt.xlabel(column)
    plt.ylabel("Nombre de produits")
    plt.show()

# Graphique de prix en fonction des categories
def plot_price_by_category(df, category_col):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x=category_col, y="price")
    plt.xticks(rotation=45)
    plt.title(f"Distribution du prix par {category_col}")
    plt.xlabel(category_col)
    plt.ylabel("Prix")
    plt.tight_layout()
    plt.show()



