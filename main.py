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

# Model de Random Forest 
from sklearn.ensemble import RandomForestRegressor

#Modelisation avec XGBoost
from xgboost import XGBRegressor

sys.path.append(os.path.abspath("./src"))

# Chargement du dataset
from load_data import load_dior

df = load_dior()
print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# Preprocessing
from preprocessing import preprocess_dior

from evaluate import(
    evaluate_model
)

from modelisation import (
    split_data,
    transform_target,
    ohe_encoding,
    count_encoding,
    target_encoding,
    train_model
)

# -----------------------

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

# Splitting
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# transformation en log
y_train_transformed, y_test_transformed, transform_info = transform_target(
    y_train, 
    y_test, 
    method='log'
)

# -----------------------

# Encodage
# Encodage Linear Regression
X_train_encoder_LR, X_test_encoder_LR, enc1 = ohe_encoding(X_train, X_test, "category1_code")
X_train_encoder_LR, X_test_encoder_LR, enc2 = count_encoding(X_train_encoder_LR, X_test_encoder_LR, "category2_code")
X_train_encoder_LR, X_test_encoder_LR, enc3 = count_encoding(X_train_encoder_LR, X_test_encoder_LR, "category3_code")

# Encodage RandomForest
X_train_encoder_RF, X_test_encoder_RF, enc4 = ohe_encoding(X_train, X_test, "category1_code")
X_train_encoder_RF, X_test_encoder_RF, enc5 = count_encoding(X_train_encoder_RF, X_test_encoder_RF, "category2_code")
X_train_encoder_RF, X_test_encoder_RF, enc6 = count_encoding(X_train_encoder_RF, X_test_encoder_RF, "category3_code")

# Encodage Gradient Boost
X_train_encoder_GB, X_test_encoder_GB, enc7 = count_encoding(X_train, X_test, "category1_code")
X_train_encoder_GB, X_test_encoder_GB, enc8 = count_encoding(X_train_encoder_GB, X_test_encoder_GB, "category2_code")
X_train_encoder_GB, X_test_encoder_GB, enc9 = count_encoding(X_train_encoder_GB, X_test_encoder_GB, "category3_code")


# verifications LR
print(X_train_encoder_LR.columns[:20])
print(X_train_encoder_LR.shape)
print(X_test_encoder_LR.shape)

# Verification RF
print(X_train_encoder_RF.columns[:20])
print(X_train_encoder_RF.shape)
print(X_test_encoder_RF.shape)

# Verification GB
print(X_train_encoder_GB.columns[:20])
print(X_train_encoder_GB.shape)
print(X_test_encoder_GB.shape)

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

# Entrainement, Test, prediction
model_RF = RandomForestRegressor(oob_score=True)

# Training 
model_RF_trained = train_model(
    model_RF, 
    X_train_encoder_RF, 
    y_train_transformed, 
    X_test_encoder_RF, 
    y_test_transformed
)

# Entrainement, Test, prediction
model_GB = XGBRegressor()

# Training 
model_GB_trained = train_model(
    model_GB, 
    X_train_encoder_GB, 
    y_train_transformed, 
    X_test_encoder_GB, 
    y_test_transformed
)

# -----------------------

# Evaluate

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
print("\n Résumé des Performances - TRAIN - Linear Regression")
print("TRAIN RMSE :", rmse_train)
print("TRAIN MAE  :", mae_train)
print("TRAIN R²   :", r2_train)

evaluation_results_LR = evaluate_model(
    model=model_LR_trained,
    X_test_encoder=X_test_encoder_LR,
    y_test_raw=y_test  
)

print("\n Résumé des Performances - TEST - Linear Regression")
print(f"Modèle : {evaluation_results_LR['model_name']}")
print(f"RMSE : {evaluation_results_LR['rmse']:,.2f}")
print(f"MAE  : {evaluation_results_LR['mae']:,.2f}")
print(f"R²   : {evaluation_results_LR['r2']:.4f}")

# ------

# Random Forest

# Train 
print(model_RF_trained.oob_score_)
# Prédictions sur le train
y_train_pred_log_RF = model_RF_trained.predict(X_train_encoder_RF)

# Inverse log transform
y_train_pred_RF = np.expm1(y_train_pred_log_RF)

# Calcul des métriques
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred_log_RF))
mae_train = mean_absolute_error(y_train, y_train_pred_RF)
r2_train = r2_score(y_train, y_train_pred_RF)

print("\n Résumé des Performances - TRAIN Random Forest")
print("TRAIN RMSE :", rmse_train)
print("TRAIN MAE  :", mae_train)
print("TRAIN R²   :", r2_train)

evaluation_results_RF = evaluate_model(
    model=model_RF_trained,
    X_test_encoder=X_test_encoder_RF,
    y_test_raw=y_test  
)

print("\n Résumé des Performances - TEST Random Forest")
print(f"Modèle : {evaluation_results_RF['model_name']}")
print(f"RMSE : {evaluation_results_RF['rmse']:,.2f}")
print(f"MAE  : {evaluation_results_RF['mae']:,.2f}")
print(f"R²   : {evaluation_results_RF['r2']:.4f}")

# ------

# Gradient Boost

# Prédictions sur le train
y_train_pred_log_GB = model_GB_trained.predict(X_train_encoder_GB)

# Inverse log transform
y_train_pred_GB = np.expm1(y_train_pred_log_GB)

# Calcul des métriques
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred_log_GB))
mae_train = mean_absolute_error(y_train, y_train_pred_GB)
r2_train = r2_score(y_train, y_train_pred_GB)

print("\n Résumé des Performances - TRAIN Gradient Boost")
print("TRAIN RMSE :", rmse_train)
print("TRAIN MAE  :", mae_train)
print("TRAIN R²   :", r2_train)

evaluation_results_GB = evaluate_model(
    model=model_GB_trained,
    X_test_encoder=X_test_encoder_GB,
    y_test_raw=y_test  
)

print("\n Résumé des Performances - TEST Gradient Boost")
print(f"Modèle : {evaluation_results_GB['model_name']}")
print(f"RMSE : {evaluation_results_GB['rmse']:,.2f}")
print(f"MAE  : {evaluation_results_GB['mae']:,.2f}")
print(f"R²   : {evaluation_results_GB['r2']:.4f}")
