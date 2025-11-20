def check_unique_values(df):
    """
    Affiche le nombre de valeurs uniques par colonne dans un DataFrame.
    """
    print("Nombre de valeurs uniques par colonne:\n")
    for col in df.columns:
        print(f"{col:20s} -> {df[col].nunique()} valeurs uniques")
