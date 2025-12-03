####################
# MAIN.PY
####################

# ID

nom = "Makoundou"
prenom = "Nsonde Chloe"
stud_numb = ""

print(nom)
print(prenom)
print(stud_numb)

# -----------------------

# Imports
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Model de regression lineaire 
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath("../src"))

# from utils_eda import (
#     check_unique_values,
#     plot_price_distribution,
#     plot_category_counts,
#     plot_price_by_category
# )

# Chargement du dataset
from load_data import load_dior

df = load_dior()
print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# -----------------------

# Preprocessing
from preprocessing import preprocess_dior

# Nettoyer et séparer features et targets
X, y = preprocess_dior(df)

# X -> title + categories
# y -> price et price_eur

# Verification
print("Taille de X:", X.shape)
print("Taille de y:", y.shape)

print("Colonnes X :", X.columns.tolist())
print("Colonnes y :", y.columns.tolist())


# -----------------------

# modelisation

from modelisation import (
    split_data,
    plot_train_test_split,
    transform_target,
    plot_target_transformation,
    ohe_encoding,
    count_encoding,
    target_encoding,
    train_model
)

# Splitting
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
plot_train_test_split(y_train, y_test)

# transformation en log
y_train_transformed, y_test_transformed, transform_info = transform_target(
    y_train, 
    y_test, 
    method='log'
)

plot_target_transformation(y_train, 'log')

# Encodage
# Encodage Linear Regression

# Application du encoding sur Category1, 2 et 3_code.

X_train_encoder_LR, X_test_encoder_LR, enc1 = ohe_encoding(X_train, X_test, "category1_code")
X_train_encoder_LR, X_test_encoder_LR, enc2 = count_encoding(X_train_encoder_LR, X_test_encoder_LR, "category2_code")
X_train_encoder_LR, X_test_encoder_LR, enc2 = count_encoding(X_train_encoder_LR, X_test_encoder_LR, "category3_code")

# verification

print(X_train_encoder_LR.columns[:20])
print(X_train_encoder_LR.shape)
print(X_test_encoder_LR.shape)

# Entrainement, Test, prediction
model_LR = LinearRegression()

# Training 
model_LR_trained = train_model(
    model_LR, 
    X_train_encoder_LR, 
    y_train_transformed, 
    X_test_encoder_LR, 
    y_test_transformed
)

# -----------------------

# Evaluate

from evaluate import(
    evaluate_model,
    plot_actual_vs_predicted,
    plot_residuals_distribution,
    plot_residuals_vs_predicted,
    plot_learning_curve,
    plot_correlation_matrix,
    plot_prediction_curve
)

# Regression Lineaire
# Prédictions sur le train
y_train_pred_log_LR = model_LR_trained.predict(X_train_encoder_LR)

# Inverse log transform
y_train_pred_LR = np.expm1(y_train_pred_log_LR)

# Calcul des métriques
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred_LR))
mae_train = mean_absolute_error(y_train, y_train_pred_LR)
r2_train = r2_score(y_train, y_train_pred_LR)

# Train
print("\n Résumé des Performances - TRAIN")
print("TRAIN RMSE :", rmse_train)
print("TRAIN MAE  :", mae_train)
print("TRAIN R²   :", r2_train)

evaluation_results = evaluate_model(
    model=model_LR_trained,
    X_test_encoder=X_test_encoder_LR,
    y_test_raw=y_test  
)

print("\n Résumé des Performances - TEST")
print(f"Modèle : {evaluation_results['model_name']}")
print(f"RMSE : {evaluation_results['rmse']:,.2f}")
print(f"MAE  : {evaluation_results['mae']:,.2f}")
print(f"R²   : {evaluation_results['r2']:.4f}")