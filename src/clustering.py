import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import os

CLUSTERING_FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'is_rush_hour', 'is_offpeak',
    'temp_c', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_severity'
]

CLUSTER_NAMES = {
    0: "Late-Night Free Flow",
    1: "Morning Commuter Peak",
    2: "Midday Commercial Transit",
    3: "Evening Rush Gridlock",
    4: "Weekend Leisure / Off-Peak"
}

def compute_elbow_silhouette(X_scaled, k_range=range(2, 8), sample_size=4000):
    """
    Computes Inertia and Silhouette scores across K values for the elbow method.
    """
    inertias = []
    silhouettes = []
    subsample = X_scaled[:sample_size] if len(X_scaled) > sample_size else X_scaled

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(subsample)
        inertias.append(float(km.inertia_))
        sil_score = float(silhouette_score(subsample, km.labels_))
        silhouettes.append(sil_score)

    return list(k_range), inertias, silhouettes

class TrafficClusterPipeline:
    """
    Hybrid Clustering Pipeline combining:
    1. K-Means for structural traffic pattern regime discovery
    2. DBSCAN for spatial density anomaly and atypical weather-traffic disruption detection
    """
    def __init__(self, n_clusters=4, dbscan_eps=1.8, dbscan_min_samples=15):
        self.n_clusters = n_clusters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=42)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        self.dbscan_nn = None
        self.cluster_profiles = {}
        self.cluster_names = {}

    def fit(self, df):
        X = df[CLUSTERING_FEATURES].values
        X_scaled = self.scaler.fit_transform(X)

        # 1. Fit K-Means
        print(f"[Clustering] Fitting K-Means with k={self.n_clusters}...")
        kmeans_labels = self.kmeans.fit_predict(X_scaled)

        # 2. Fit DBSCAN on a representative sample if dataset is large, then train nearest neighbors
        print(f"[Clustering] Fitting DBSCAN (eps={self.dbscan_eps}, min_samples={self.dbscan_min_samples})...")
        sample_n = min(12000, len(X_scaled))
        indices = np.random.RandomState(42).choice(len(X_scaled), sample_n, replace=False)
        X_db_sample = X_scaled[indices]
        db_sample_labels = self.dbscan.fit_predict(X_db_sample)

        # Fit 1-NN on DBSCAN core samples to allow out-of-sample prediction
        self.dbscan_nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        self.dbscan_nn.fit(X_db_sample)
        self.db_sample_labels = db_sample_labels

        # Fit 2D PCA for interactive visual projection
        self.pca.fit(X_scaled)

        # 3. Build cluster profiles and descriptive names
        df_copy = df.copy()
        df_copy['kmeans_cluster'] = kmeans_labels
        
        # Determine descriptive names dynamically based on average volume and rush hours
        cluster_summary = df_copy.groupby('kmeans_cluster').agg({
            'traffic_volume': ['mean', 'std', 'min', 'max'],
            'hour': 'mean',
            'is_weekend': 'mean',
            'is_rush_hour': 'mean'
        })
        
        # Order clusters by mean traffic volume to assign clean intuitive designations
        mean_vols = df_copy.groupby('kmeans_cluster')['traffic_volume'].mean().to_dict()
        sorted_clusters = sorted(mean_vols.keys(), key=lambda c: mean_vols[c])
        
        default_names = [
            "Late Night Free Flow (Low Volume)",
            "Off-Peak / Weekend Leisure Flow",
            "Midday Steady Transit (Moderate)",
            "Peak Commuter Congestion (High)",
            "Severe Bottleneck Gridlock"
        ]
        
        self.cluster_names = {}
        for rank, c_id in enumerate(sorted_clusters):
            self.cluster_names[c_id] = default_names[min(rank, len(default_names)-1)]

        self.cluster_profiles = cluster_summary.to_dict()
        print("[Clustering] Clustering models fitted successfully.")
        return self

    def transform(self, df):
        """
        Transforms input DataFrame into cluster labels and anomaly flags for hybrid feature feeding.
        """
        X = df[CLUSTERING_FEATURES].values
        X_scaled = self.scaler.transform(X)

        # K-Means cluster assignment
        kmeans_labels = self.kmeans.predict(X_scaled)

        # DBSCAN anomaly assignment via nearest core point
        distances, indices = self.dbscan_nn.kneighbors(X_scaled)
        # If distance is greater than eps or nearest point was noise (-1), mark as anomaly
        nearest_labels = self.db_sample_labels[indices.flatten()]
        is_anomaly = ((distances.flatten() > self.dbscan_eps) | (nearest_labels == -1)).astype(int)

        # PCA coordinates for visualization
        pca_coords = self.pca.transform(X_scaled)

        return kmeans_labels, is_anomaly, pca_coords

    def get_cluster_name(self, cluster_id):
        return self.cluster_names.get(cluster_id, f"Traffic Regime #{cluster_id}")
