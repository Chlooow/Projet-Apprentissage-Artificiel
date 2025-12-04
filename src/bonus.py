import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def run_clustering(X_features, df_analysis, n_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    # algo des kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)

    df_analysis['cluster_id'] = cluster_labels

    cluster_summary = df_analysis.groupby('cluster_id')['price_original'].agg(['mean', 'count'])
    cluster_summary = cluster_summary.rename(columns={'mean': 'Prix Moyen', 'count': 'Taille du Segment'})

    return cluster_summary, pd.Series(cluster_labels, index=X_features.index), kmeans, scaler

def run_segmented_regression(X_train, y_train, X_test, y_test, cluster_labels_train, kmeans_model, scaler, n_estimators=100, max_depth=15):
    X_train_segmented = X_train.copy()
    y_train_segmented = y_train.copy()
    X_train_segmented['cluster_id'] = cluster_labels_train
    
    segmented_models = {}
    
    for cluster_id in cluster_labels_train.unique():
        X_segment = X_train_segmented[X_train_segmented['cluster_id'] == cluster_id].drop(columns=['cluster_id'])
        y_segment = y_train_segmented.loc[X_segment.index]
        
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=42, 
            max_depth=max_depth, 
            n_jobs=-1
        )
        rf_model.fit(X_segment, y_segment.values.ravel())
        segmented_models[cluster_id] = rf_model

    X_test_scaled = scaler.transform(X_test)
    test_cluster_labels = kmeans_model.predict(X_test_scaled)

    X_test_segmented = X_test.copy()
    X_test_segmented['cluster_id'] = test_cluster_labels
    y_test_pred_final = pd.Series(index=y_test.index, dtype=float)

    # Boucle de pred
    for cluster_id in segmented_models.keys():
        X_test_segment = X_test_segmented[X_test_segmented['cluster_id'] == cluster_id].drop(columns=['cluster_id'])
        
        if len(X_test_segment) > 0:
            model = segmented_models[cluster_id]
            y_pred_test_segment = model.predict(X_test_segment)
            y_test_pred_final.loc[X_test_segment.index] = y_pred_test_segment
    r2_test = r2_score(y_test, y_test_pred_final)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred_final))
    mae_test = mean_absolute_error(y_test, y_test_pred_final)

    results = {
        'R2': r2_test,
        'RMSE': rmse_test,
        'MAE': mae_test
    }

    return results, segmented_models