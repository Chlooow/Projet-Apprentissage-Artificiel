import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_clustering(X_features, df_analysis, n_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    # algo des kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)

    df_analysis['cluster_id'] = cluster_labels

    cluster_summary = df_analysis.groupby('cluster_id')['price_original'].agg(['mean', 'count'])
    cluster_summary = cluster_summary.rename(columns={'mean': 'Prix Moyen', 'count': 'Taille du Segment'})

    return cluster_summary, pd.Series(cluster_labels, index=X_features.index), kmeans