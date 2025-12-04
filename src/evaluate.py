import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor

def evaluate_model(model, X_test_encoder, y_test_raw):
    y_pred_log = model.predict(X_test_encoder)

    # inversion de la transformation pour evaluer
    y_pred = np.expm1(y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))

    mae = mean_absolute_error(y_test_raw, y_pred)
    r2 = r2_score(y_test_raw, y_pred)

    return {
        'model_name': model.__class__.__name__,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred_original': y_pred
    }

def run_segmented_regression(X_train_segmented, X_test_segmented, y_train_transformed, y_test, n_clusters=3, model_class=RandomForestRegressor):
    models_segmented = {}
    y_test_preds_list = []
    y_test_raw_list = []

    final_features = X_train_segmented.columns.drop(['cluster_id'])
    for k in range(n_clusters):
        X_train_k = X_train_segmented[X_train_segmented['cluster_id'] == k].drop(columns=['cluster_id'])
        y_train_k_log = y_train_transformed.loc[X_train_k.index]

        if X_train_k.empty:
            print(f"ATTENTION: Le Segment {k} est vide dans le jeu d'ENTRAÎNEMENT. Ignoré.")
            continue

        model_k = model_class(n_estimators=100, random_state=42, n_jobs=-1)
        model_k.fit(X_train_k, y_train_k_log.values.ravel()) 
        models_segmented[k] = model_k

        X_test_k = X_test_segmented[X_test_segmented['cluster_id'] == k].drop(columns=['cluster_id'])
        y_test_k = y_test.loc[X_test_k.index]

        if X_test_k.empty:
            print(f"ATTENTION: Le Segment {k} est vide dans le jeu de TEST. Skippé pour l'évaluation.")
            continue # Passe à l'itération suivante

        evaluation_results_k = evaluate_model(
            model=model_k, 
            X_test_encoder=X_test_k,
            y_test_raw=y_test_k
        )

        y_pred_original_k = pd.Series(evaluation_results_k['y_pred_original'], index=y_test_k.index)
        y_test_preds_list.append(y_pred_original_k)
        y_test_raw_list.append(y_test_k)

        # Affichage pour le suivi
        print(f"RMSE Segment {k} : {evaluation_results_k['rmse']:,.2f} (Taille: {len(X_test_k)} échantillons)")

    y_test_preds_segmented = pd.concat(y_test_preds_list).sort_index()
    y_test_raw_global = pd.concat(y_test_raw_list).sort_index()

    return models_segmented, y_test_preds_segmented, y_test_raw_global, final_features

def plot_actual_vs_predicted(y_test_original, y_pred, model_name):
    plt.figure(figsize=(10, 6))

    x_data = y_test_original.values.flatten()
    y_data = y_pred.flatten()
    
    # Nuage de points
    sns.scatterplot(x=x_data, y=y_data)
    
    # Ligne de la "prédiction parfaite" (y=x)
    max_val = max(x_data.max(), y_data.max())
    min_val = min(x_data.min(), y_data.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ajustement Parfait')
    
    plt.title(f'Valeurs Réelles vs. Prédites ({model_name})')
    plt.xlabel('Prix Réel (CNY)')
    plt.ylabel('Prix Prédit (CNY)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def plot_residuals_distribution(y_test_original, y_pred, model_name):
    residuals = y_test_original.values.flatten() - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=50)
    plt.axvline(x=0, color='red', linestyle='--', label='Moyenne = 0')
    
    plt.title(f'Distribution des Résidus ({model_name})')
    plt.xlabel('Résidus (Erreur de Prédiction)')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def plot_residuals_vs_predicted(y_pred, y_test_original, model_name):
    
    y_data = y_pred.flatten()
    residuals = y_test_original.values.flatten() - y_data

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_data, y=residuals)
    plt.axhline(y=0, color='red', linestyle='--', label='Zéro Résidu')
    
    plt.title(f'Résidus vs. Valeurs Prédites ({model_name})')
    plt.xlabel('Prix Prédit (CNY)')
    plt.ylabel('Résidus (Erreur)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def plot_learning_curve(model, X, y_log, cv=5, scoring='neg_mean_squared_error'):

    """
    Génère et affiche la courbe d'apprentissage d'un modèle.
    """
    
    # Générer les données pour la courbe d'apprentissage
    train_sizes, train_scores, test_scores = learning_curve(
        model, 
        X, 
        y_log.values.flatten(), # Utiliser la cible log-transformée
        cv=cv, 
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(.1, 1.0, 10) # Utiliser 10 fractions du jeu d'entraînement
    )

    # Calculer les moyennes et les écarts types
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # La métrique 'neg_mean_squared_error' est négative, on prend la racine carrée 
    # et on inverse le signe pour obtenir le RMSE positif sur l'échelle log
    train_errors_mean = np.sqrt(-train_scores_mean)
    test_errors_mean = np.sqrt(-test_scores_mean)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.grid()

    plt.title(f"Courbe d'Apprentissage ({model.__class__.__name__})")
    plt.xlabel("Taille de l'échantillon d'entraînement")
    plt.ylabel("RMSE (sur échelle Log)")
    
    # Courbe d'entraînement
    plt.plot(train_sizes, train_errors_mean, 'o-', color="r", label="Erreur d'entraînement")
    # Courbe de validation croisée
    plt.plot(train_sizes, test_errors_mean, 'o-', color="g", label="Erreur de validation croisée")

    # Affichage des écarts types (facultatif)
    plt.fill_between(train_sizes, train_errors_mean - train_scores_std,
                     train_errors_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_errors_mean - test_scores_std,
                     test_errors_mean + test_scores_std, alpha=0.1, color="g")

    plt.legend(loc="best")
    plt.show()

def plot_correlation_matrix(X_data, y_data_log, title="Matrice de Corrélation des Features"):
    """
    Calcule et affiche la matrice de corrélation entre toutes les features 
    et la cible transformée.
    """
    
    # 1. Créer le DataFrame combiné pour le calcul de corrélation
    # Assurez-vous que les indices correspondent avant de concaténer
    df_combined = X_data.copy()
    df_combined['price_log'] = y_data_log.values 

    # 2. Calculer la matrice de corrélation (méthode de Pearson)
    correlation_matrix = df_combined.corr()

    # 3. Afficher la heatmap
    plt.figure(figsize=(14, 12))
    
    # Utilisation d'un masque si la matrice est très grande (optionnel)
    mask = np.triu(correlation_matrix)
    
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        mask=mask, # Afficher seulement la partie inférieure
        cbar=True,
        linewidths=.5,
        linecolor='black'
    )
    
    plt.title(title, fontsize=16)
    plt.show()


def plot_prediction_curve(y_test_original, y_pred, model_name, sample_size=100):
    """
    Affiche les prix réels vs. les prix prédits sur une séquence de l'échantillon de test.
    
    Args:
        y_test_original (pd.Series): Prix réels (non transformés).
        y_pred (np.array): Prix prédits (non transformés).
        model_name (str): Nom du modèle.
        sample_size (int): Nombre de points à afficher séquentiellement.
    """
    
    # Assurer que les prédictions sont un tableau 1D
    y_pred_flat = y_pred.flatten() 
    
    # Assurer que la cible réelle est un tableau 1D
    y_test_flat = y_test_original.values.flatten()

    # Choisir un segment pour la clarté (par exemple, les 100 premiers échantillons)
    x_indices = np.arange(sample_size)
    
    plt.figure(figsize=(15, 6))
    
    # Tracer les valeurs réelles
    plt.plot(x_indices, y_test_flat[:sample_size], label='Prix Réel', color='blue', alpha=0.7)
    
    # Tracer les valeurs prédites
    plt.plot(x_indices, y_pred_flat[:sample_size], label='Prix Prédit', color='red', linestyle='--')
    
    plt.title(f'Courbe de Prédiction sur Échantillon Séquentiel ({model_name})')
    plt.xlabel('Index de l\'Échantillon de Test')
    plt.ylabel('Prix (CNY)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
