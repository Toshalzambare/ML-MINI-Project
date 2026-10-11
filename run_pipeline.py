import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_clean_data
from src.feature_engineering import (
    engineer_features, BASE_FEATURE_COLS, HYBRID_FEATURE_COLS,
    CONGESTION_BINARY_LABELS, CONGESTION_TIER_LABELS
)
from src.clustering import TrafficClusterPipeline, compute_elbow_silhouette, CLUSTERING_FEATURES
from src.modeling import HybridTrafficModel
from src.evaluate import generate_evaluation_artifacts

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def run_complete_pipeline():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 70)
    print("STEP 1: DATA INGESTION & DATA CLEANING")
    print("=" * 70)
    raw_df = load_clean_data()
    print(f"Data successfully loaded. Shape: {raw_df.shape}")

    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING & TEMPORAL/WEATHER/LAG DECOMPOSITION")
    print("=" * 70)
    df, encoders = engineer_features(raw_df, is_training=True)
    print(f"Engineered features ready. Total columns: {len(df.columns)}")

    print("\n" + "=" * 70)
    print("STEP 3: TRAFFIC FLOW PATTERN CLUSTERING (K-MEANS & DBSCAN)")
    print("=" * 70)
    # 1. Elbow and Silhouette analysis
    X_clust = df[CLUSTERING_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)
    k_range, inertias, silhouettes = compute_elbow_silhouette(X_scaled, k_range=range(2, 7), sample_size=3500)
    
    clustering_eval_data = {
        'k_range': k_range,
        'inertias': inertias,
        'silhouettes': silhouettes
    }

    # 2. Fit clustering pipeline
    cluster_pipeline = TrafficClusterPipeline(n_clusters=4, dbscan_eps=1.8, dbscan_min_samples=15)
    cluster_pipeline.fit(df)
    
    # 3. Transform dataset with cluster regime & anomaly flags
    kmeans_labels, dbscan_anomalies, pca_coords = cluster_pipeline.transform(df)
    df['cluster_regime'] = kmeans_labels
    df['is_density_anomaly'] = dbscan_anomalies
    df['pca_x'] = pca_coords[:, 0]
    df['pca_y'] = pca_coords[:, 1]
    
    anomaly_count = int(df['is_density_anomaly'].sum())
    print(f"K-Means clusters discovered: {cluster_pipeline.n_clusters}")
    print(f"DBSCAN density anomalies identified: {anomaly_count} ({anomaly_count/len(df)*100:.2f}%)")

    clustering_eval_data['cluster_names'] = cluster_pipeline.cluster_names
    clustering_eval_data['anomaly_rate'] = float(anomaly_count / len(df))

    print("\n" + "=" * 70)
    print("STEP 4: HYBRID PREDICTIVE MODELING (RANDOM FOREST + GRADIENT BOOSTING)")
    print("=" * 70)
    X = df[HYBRID_FEATURE_COLS].values
    y_bin = df['congestion_binary'].values
    y_tier = df['congestion_tier'].values
    y_reg = df['traffic_volume'].values

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train_b, y_test_b, y_train_t, y_test_t, y_train_r, y_test_r = train_test_split(
        X, y_bin, y_tier, y_reg, test_size=0.20, random_state=42, stratify=y_bin
    )

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    hybrid_model = HybridTrafficModel(random_state=42)
    hybrid_model.fit(X_train, y_train_b, y_train_t, y_train_r, feature_names=HYBRID_FEATURE_COLS)

    print("\n" + "=" * 70)
    print("STEP 5: COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    cls_metrics, reg_metrics, tier_acc, tier_f1 = hybrid_model.evaluate_all(X_test, y_test_b, y_test_t, y_test_r)
    
    print("\n--- PRIMARY CLASSIFICATION PERFORMANCE (CONGESTED VS FREE-FLOW) ---")
    for model_name, metrics in cls_metrics.items():
        print(f"  [{model_name}] Accuracy: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1_weighted']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

    print(f"\n--- 3-TIER LEVEL OF SERVICE ACCURACY: {tier_acc*100:.2f}% | F1: {tier_f1:.4f} ---")

    print("\n--- REGRESSION PERFORMANCE (HOURLY TRAFFIC VOLUME) ---")
    for model_name, metrics in reg_metrics.items():
        print(f"  [{model_name}] R2: {metrics['r2']:.4f} | MAE: {metrics['mae']:.1f} vph | RMSE: {metrics['rmse']:.1f} vph")

    # Generate Confusion Matrix and Classification Report for the Hybrid Model
    hybrid_test_preds, _, _ = hybrid_model.predict_binary(X_test)
    cm_data, report_data = generate_evaluation_artifacts(y_test_b, hybrid_test_preds, CONGESTION_BINARY_LABELS, MODELS_DIR)

    # Feature Importance
    fi_df = hybrid_model.get_feature_importances()
    fi_df.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)

    print("\n" + "=" * 70)
    print("STEP 6: PERSISTING MODELS & METADATA ARTIFACTS")
    print("=" * 70)
    joblib.dump(cluster_pipeline, os.path.join(MODELS_DIR, "cluster_pipeline.joblib"))
    joblib.dump(hybrid_model, os.path.join(MODELS_DIR, "hybrid_model.joblib"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.joblib"))

    with open(os.path.join(MODELS_DIR, "clustering_summary.json"), 'w') as f:
        json.dump(clustering_eval_data, f, indent=2)

    with open(os.path.join(MODELS_DIR, "metrics_summary.json"), 'w') as f:
        json.dump({
            "classification": cls_metrics,
            "regression": reg_metrics,
            "tier_accuracy": tier_acc,
            "tier_f1": tier_f1,
            "hybrid_accuracy": cls_metrics["Hybrid Ensemble"]["accuracy"],
            "hybrid_f1": cls_metrics["Hybrid Ensemble"]["f1_weighted"],
            "hybrid_r2": reg_metrics["Hybrid Ensemble"]["r2"],
            "hybrid_mae": reg_metrics["Hybrid Ensemble"]["mae"],
            "total_records": len(df),
            "test_records": len(X_test)
        }, f, indent=2)

    # Lightweight sample for Streamlit fast exploration
    sample_df = df.sample(min(8000, len(df)), random_state=42)
    sample_df.to_csv(os.path.join(MODELS_DIR, "sample_data.csv"), index=False)

    print(f"\nAll models, encoders, and artifacts saved to: {MODELS_DIR}")
    print("Pipeline execution complete! Ready for Streamlit UI launch.")

if __name__ == "__main__":
    run_complete_pipeline()
