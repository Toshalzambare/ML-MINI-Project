import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

class HybridTrafficModel:
    """
    Hybrid Predictive Architecture combining:
    - Random Forest: Bagging variance reduction
    - Gradient Boosting: Sequential bias reduction
    - Integrated Unsupervised Cluster Regime & DBSCAN Anomaly Inputs
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Primary Binary Congestion Classifier (Free Flow vs Congested)
        self.rf_classifier_bin = RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        )
        self.gb_classifier_bin = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=12,
            learning_rate=0.08,
            random_state=random_state
        )

        # Multi-Tier Congestion Classifier (3-Tier Level-of-Service)
        self.gb_classifier_tier = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=12,
            learning_rate=0.08,
            random_state=random_state
        )
        
        # Continuous Traffic Volume Regressors
        self.rf_regressor = RandomForestRegressor(
            n_estimators=120,
            max_depth=16,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        )
        self.gb_regressor = HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=12,
            learning_rate=0.08,
            random_state=random_state
        )

        self.feature_names = []

    def fit(self, X_train, y_train_bin, y_train_tier, y_train_reg, feature_names=None):
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
        
        print("[Modeling] Fitting Random Forest Binary Classifier...")
        self.rf_classifier_bin.fit(X_train, y_train_bin)
        
        print("[Modeling] Fitting Gradient Boosting Binary Classifier...")
        self.gb_classifier_bin.fit(X_train, y_train_bin)

        print("[Modeling] Fitting Gradient Boosting Multi-Tier Classifier...")
        self.gb_classifier_tier.fit(X_train, y_train_tier)
        
        print("[Modeling] Fitting Random Forest Regressor...")
        self.rf_regressor.fit(X_train, y_train_reg)
        
        print("[Modeling] Fitting Gradient Boosting Regressor...")
        self.gb_regressor.fit(X_train, y_train_reg)

        print("[Modeling] All supervised models fitted successfully.")
        return self

    def predict_binary(self, X):
        """
        Predicts Congestion State (0: Free-Flow, 1: Congested) via Hybrid Soft-Voting.
        """
        prob_rf = self.rf_classifier_bin.predict_proba(X)
        prob_gb = self.gb_classifier_bin.predict_proba(X)
        
        prob_hybrid = 0.45 * prob_rf + 0.55 * prob_gb
        pred_cls = np.argmax(prob_hybrid, axis=1)
        confidences = np.max(prob_hybrid, axis=1)
        
        return pred_cls, confidences, prob_hybrid

    def predict_tier(self, X):
        """
        Predicts 3-Tier Congestion Level (0: Low, 1: Moderate, 2: Severe).
        """
        probs = self.gb_classifier_tier.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    def predict_regression(self, X):
        """
        Predicts continuous Traffic Volume using Hybrid Blending (0.45 RF + 0.55 GB).
        """
        pred_rf = self.rf_regressor.predict(X)
        pred_gb = self.gb_regressor.predict(X)
        pred_hybrid = 0.45 * pred_rf + 0.55 * pred_gb
        return np.round(pred_hybrid, 1)

    def evaluate_all(self, X_test, y_test_bin, y_test_tier, y_test_reg):
        """
        Evaluates Random Forest, Gradient Boosting, and Hybrid Ensemble.
        """
        # 1. Binary Congestion Evaluation
        rf_bin_preds = self.rf_classifier_bin.predict(X_test)
        gb_bin_preds = self.gb_classifier_bin.predict(X_test)
        hybrid_bin_preds, _, _ = self.predict_binary(X_test)

        cls_results = {
            "Random Forest": {
                "accuracy": float(accuracy_score(y_test_bin, rf_bin_preds)),
                "f1_weighted": float(f1_score(y_test_bin, rf_bin_preds, average='weighted')),
                "precision": float(precision_score(y_test_bin, rf_bin_preds, average='weighted')),
                "recall": float(recall_score(y_test_bin, rf_bin_preds, average='weighted'))
            },
            "Gradient Boosting": {
                "accuracy": float(accuracy_score(y_test_bin, gb_bin_preds)),
                "f1_weighted": float(f1_score(y_test_bin, gb_bin_preds, average='weighted')),
                "precision": float(precision_score(y_test_bin, gb_bin_preds, average='weighted')),
                "recall": float(recall_score(y_test_bin, gb_bin_preds, average='weighted'))
            },
            "Hybrid Ensemble": {
                "accuracy": float(accuracy_score(y_test_bin, hybrid_bin_preds)),
                "f1_weighted": float(f1_score(y_test_bin, hybrid_bin_preds, average='weighted')),
                "precision": float(precision_score(y_test_bin, hybrid_bin_preds, average='weighted')),
                "recall": float(recall_score(y_test_bin, hybrid_bin_preds, average='weighted'))
            }
        }

        # Multi-Tier evaluation
        tier_preds, _ = self.predict_tier(X_test)
        tier_acc = float(accuracy_score(y_test_tier, tier_preds))
        tier_f1 = float(f1_score(y_test_tier, tier_preds, average='weighted'))

        # 2. Regression Evaluation
        rf_reg_preds = self.rf_regressor.predict(X_test)
        gb_reg_preds = self.gb_regressor.predict(X_test)
        hybrid_reg_preds = self.predict_regression(X_test)

        reg_results = {
            "Random Forest": {
                "r2": float(r2_score(y_test_reg, rf_reg_preds)),
                "mae": float(mean_absolute_error(y_test_reg, rf_reg_preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_test_reg, rf_reg_preds)))
            },
            "Gradient Boosting": {
                "r2": float(r2_score(y_test_reg, gb_reg_preds)),
                "mae": float(mean_absolute_error(y_test_reg, gb_reg_preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_test_reg, gb_reg_preds)))
            },
            "Hybrid Ensemble": {
                "r2": float(r2_score(y_test_reg, hybrid_reg_preds)),
                "mae": float(mean_absolute_error(y_test_reg, hybrid_reg_preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_test_reg, hybrid_reg_preds)))
            }
        }

        return cls_results, reg_results, tier_acc, tier_f1

    def get_feature_importances(self):
        importances = self.rf_classifier_bin.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        return fi_df
