from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import pandas as pd

def choose_encoder(df, encoding_config: dict, target=None):
    df = df.copy()
    encoders = {}

    for col, method in encoding_config.items():
        if method is None:
            continue
        method = method.lower()
        if method == "onehot":
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            arr = ohe.fit_transform(df[[col]])
            df_ohe = pd.DataFrame(
                arr,
                columns=[f"{col}_{v}" for v in ohe.categories_[0]],
                index=df.index
            )
            df = pd.concat([df.drop(columns=[col]), df_ohe], axis=1)
            encoders[col] = ohe

        elif method == "label":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        elif method == "ordinal":
            oe = OrdinalEncoder()
            df[col] = oe.fit_transform(df[[col]])
            encoders[col] = oe

        elif method == "binaire":
            # On considère que la colonne est 0/1 ou True/False
            df[col] = df[col].astype(int)

        elif method == "target":
            if target is None:
                raise ValueError(f"Pour le target encoding, il faut fournir la colonne target.")
            # Encodage par moyenne du target
            target_map = target.groupby(df[col]).mean()
            df[col] = df[col].map(target_map)
            encoders[col] = target_map

        else:
            print(f"[WARN] Méthode '{method}' non reconnue pour '{col}' → ignorée")
            continue
    
    return df, encoders

def verify_encod(X_origin, X_encoded, cols):
    for col in cols:
        print(f"\n=== Vérification de la colonne : {col}===")

        # Avant encodage
        if col in X_origin.columns:
            print("Avant encodage :", X_origin[col].unique())
        else:
            print("Avant encodage : colonne absente de X_original")
        
        # Colonnes encodées associées
        encoded_cols = [c for c in X_encoded.columns if c.startswith(col)]

        if encoded_cols:
            print("\nAprès encodage :")
            for ec in encoded_cols:
                print(f"{ec}: {X_encoded[ec].unique()[:20]}")
        else:
            print("\nAprès encodage : aucune colonne encodée trouvée.")

