from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    print(f"Shape X_train : {X_train.shape}")
    print(f"Shape X_test  : {X_test.shape}")
    print(f"Shape y_train : {y_train.shape}")
    print(f"Shape y_test  : {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def transform_target(y_train, y_test, method='log'):
    transform_info = {'method': method}
    
    if method == 'log':
        # log1p = log(1 + x) pour éviter log(0)
        y_train_transformed = np.log1p(y_train)
        y_test_transformed = np.log1p(y_test)
        print(f"Transformation: log(1 + prix)")
        
    elif method == 'sqrt':
        y_train_transformed = np.sqrt(y_train)
        y_test_transformed = np.sqrt(y_test)
        print(f"Transformation: √prix")
        
    elif method == 'standardize':
        scaler = StandardScaler()
        y_train_transformed = scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_transformed = scaler.transform(y_test.values.reshape(-1, 1)).ravel()
        transform_info['scaler'] = scaler
        print(f"Transformation: standardisation Z-score")
        
    else:
        raise ValueError(f"Méthode '{method}' non reconnue")
    
    return y_train_transformed, y_test_transformed, transform_info

def inverse_transform_target(y_pred, transform_info):
    method = transform_info['method']
    
    if method == 'log':
        return np.expm1(y_pred)  # exp(x) - 1
    elif method == 'sqrt':
        return np.square(y_pred)
    elif method == 'standardize':
        scaler = transform_info['scaler']
        return scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    
    return y_pred

def plot_train_test_split(y_train, y_test, title="Répartition Train/Test"):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1: Distribution des prix
    axes[0].hist(y_train, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
    axes[0].hist(y_test, bins=50, alpha=0.7, label='Test', color='orange', edgecolor='black')
    axes[0].set_xlabel('Prix (CNY)', fontsize=12)
    axes[0].set_ylabel('Fréquence', fontsize=12)
    axes[0].set_title('Distribution des prix - Train vs Test', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Graphique 2: Proportions
    sizes = [len(y_train), len(y_test)]
    labels = [f'Train\n({len(y_train)} samples)\n{len(y_train)/(len(y_train)+len(y_test))*100:.1f}%',
              f'Test\n({len(y_test)} samples)\n{len(y_test)/(len(y_train)+len(y_test))*100:.1f}%']
    colors = ['#3498db', '#e74c3c']
    explode = (0.05, 0.05)
    
    axes[1].pie(sizes, labels=labels, colors=colors, autopct='', 
                startangle=90, explode=explode, shadow=True, textprops={'fontsize': 11})
    axes[1].set_title('Proportion Train/Test', fontsize=14, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_target_transformation(y, method='log'):

    if method == 'log':
        y_trans = np.log1p(y)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(y, bins=30)
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.hist(y_trans, bins=30)
    plt.title("Transformée (log)")

    plt.tight_layout()
    plt.show()

    return y_trans

def ohe_encoding(X_train, X_test, col):
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # fit_transform uniquement sur X_train
    train_encoded = encoder.fit_transform(X_train[[col]])
    test_encoded = encoder.transform(X_test[[col]])

    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

    train_df = pd.DataFrame(train_encoded, columns=feature_names, index=X_train_encoded.index)
    test_df = pd.DataFrame(test_encoded, columns=feature_names, index=X_test_encoded.index)

    # Supprimer la colonne originale
    X_train_encoded = X_train_encoded.drop(columns=[col])
    X_test_encoded = X_test_encoded.drop(columns=[col])

    X_train_encoded = pd.concat([X_train_encoded, train_df], axis=1)
    X_test_encoded = pd.concat([X_test_encoded, test_df], axis=1)
    
    return X_train_encoded, X_test_encoded, encoder

def count_encoding(X_train, X_test, col):
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    # Comptage sur le train uniquement
    counts = X_train[col].value_counts()

    # Encodage train
    X_train_encoded[col + "_count"] = X_train[col].map(counts)

    # Encodage test (catégories inconnues -> 0)
    X_test_encoded[col + "_count"] = X_test[col].map(counts).fillna(0)

    # On peut supprimer la colonne originale si tu veux
    X_train_encoded = X_train_encoded.drop(columns=[col])
    X_test_encoded = X_test_encoded.drop(columns=[col])

    return X_train_encoded, X_test_encoded, counts

def target_encoding(X_train, X_test, target, col):
    # Moyenne de la target par catégorie (calculée sur le train uniquement)
    mapping = X_train.join(target).groupby(col)[target.name].mean()

    # Remplacement dans le train
    X_train_encoded = X_train.copy()
    X_train_encoded[col] = X_train[col].map(mapping)

    # Remplacement dans le test (avec fallback sur la moyenne globale)
    global_mean = target.mean()
    X_test_encoded = X_test.copy()
    X_test_encoded[col] = X_test[col].map(mapping).fillna(global_mean)

    return X_train_encoded, X_test_encoded, mapping

def train_model(model, X_train_encoder, y_train_log, X_test_encoder, y_test):
    model.fit(X_train_encoder, y_train_log)

    return model

def run_segmented_regression(X_train_segmented, X_test_segmented, y_train_transformed, y_test, n_clusters=3, model_class):
    models_segmented = {}
    y_test_preds_list = []
    y_test_raw_list = []

    final_features = X_train_segmented.columns.drop(['cluster_id'])
    for k in range(n_clusters):
        X_train_k = X_train_segmented[X_train_segmented['cluster_id'] == k].drop(columns=['cluster_id'])
        y_train_k_log = y_train_transformed.loc[X_train_k.index]

        model_k = model_class(n_estimators=100, random_state=42, n_jobs=-1)
        model_k.fit(X_train_k, y_train_k_log.values.ravel()) 
        models_segmented[k] = model_k

        X_test_k = X_test_segmented[X_test_segmented['cluster_id'] == k].drop(columns=['cluster_id'])
        y_test_k = y_test.loc[X_test_k.index]
