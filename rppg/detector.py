"""
Deepfake Detector & Classifier Module
====================================
Implements window-based feature extraction from spatial rPPG signals,
trains an SVM/RandomForest classifier, performs GroupKFold cross-validation
(grouped by person ID), and applies window-level voting for video classification.
"""

import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from .pulse_analyzer import PulseAnalyzer
from .spatial_coherence import SpatialCoherenceAnalyzer
from .signal_processor import SignalProcessor


class DeepfakePulseDetector:
    """Classifies video segments and entire videos as REAL or DEEPFAKE based on rPPG pulse coherence."""

    FEATURE_NAMES = [
        "mean_coherence",
        "min_coherence",
        "harmonic_coherence",
        "mean_cross_correlation",
        "mean_lag_abs",
        "bpm_dispersion",
        "mean_snr_db",
        "mean_spectral_entropy",
        "bpm_in_normal_range"
    ]

    def __init__(self, classifier_type="svm"):
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        if classifier_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        else:
            self.model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        self.is_trained = False

    @classmethod
    def extract_window_features(cls, roi_clean_signals, fs):
        """
        Extracts a feature vector for a single temporal video window.
        Args:
            roi_clean_signals: dict of {roi_name: 1D np.ndarray of cleaned pulse signal}
            fs: float, sampling rate.
        Returns:
            np.ndarray of shape (len(FEATURE_NAMES),)
        """
        coherence_stats = SpatialCoherenceAnalyzer.analyze_patch_set(roi_clean_signals, fs)

        # Extended harmonic coherence (0.7 - 14.0 Hz)
        harmonic_coherences = []
        roi_names = sorted(list(roi_clean_signals.keys()))
        for i in range(len(roi_names)):
            for j in range(i + 1, len(roi_names)):
                s_i = roi_clean_signals[roi_names[i]]
                s_j = roi_clean_signals[roi_names[j]]
                h_coh, _, _ = SpatialCoherenceAnalyzer.compute_pairwise_coherence(
                    s_i, s_j, fs, lowcut=0.7, highcut=min(14.0, (fs / 2.0) * 0.95)
                )
                harmonic_coherences.append(h_coh)
        mean_harmonic_coh = float(np.mean(harmonic_coherences)) if harmonic_coherences else 0.0

        # SNR and Spectral Entropy across ROIs
        snr_list = []
        entropy_list = []
        bpms = []
        for name, sig in roi_clean_signals.items():
            hr_info = PulseAnalyzer.estimate_heart_rate(sig, fs)
            snr_list.append(hr_info["snr_db"])
            entropy_list.append(hr_info["spectral_entropy"])
            bpms.append(hr_info["bpm"])

        mean_snr = float(np.mean(snr_list)) if snr_list else 0.0
        mean_entropy = float(np.mean(entropy_list)) if entropy_list else 1.0

        median_bpm = float(np.median(bpms)) if bpms else 0.0
        # Normal resting human heart rate typically 50 - 120 BPM
        bpm_in_range = 1.0 if (50.0 <= median_bpm <= 120.0) else 0.0

        features = [
            coherence_stats["mean_coherence"],
            coherence_stats["min_coherence"],
            mean_harmonic_coh,
            coherence_stats["mean_cross_correlation"],
            coherence_stats["mean_lag_abs"],
            coherence_stats["bpm_dispersion"],
            mean_snr,
            mean_entropy,
            bpm_in_range
        ]
        return np.array(features, dtype=np.float32)

    def train(self, X, y, groups=None, n_splits=5):
        """
        Trains the classifier. If groups (person IDs) are provided, performs GroupKFold cross-validation.
        Args:
            X: np.ndarray of shape (N_samples, N_features)
            y: np.ndarray of shape (N_samples,), where 0 = REAL, 1 = DEEPFAKE
            groups: list or np.ndarray of person/video IDs to prevent identity leakage
            n_splits: int, number of cross-validation folds
        Returns:
            dict of evaluation metrics
        """
        X = np.asarray(X)
        y = np.asarray(y)

        metrics = {}
        if groups is not None and len(np.unique(groups)) >= n_splits:
            gkf = GroupKFold(n_splits=n_splits)
            accs, precs, recs, aucs = [], [], [], []
            for train_idx, val_idx in gkf.split(X, y, groups):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                fold_scaler = StandardScaler()
                X_tr_sc = fold_scaler.fit_transform(X_tr)
                X_val_sc = fold_scaler.transform(X_val)

                if self.classifier_type == "rf":
                    fold_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
                else:
                    fold_model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)

                fold_model.fit(X_tr_sc, y_tr)
                preds = fold_model.predict(X_val_sc)
                probs = fold_model.predict_proba(X_val_sc)[:, 1]

                accs.append(accuracy_score(y_val, preds))
                precs.append(precision_score(y_val, preds, zero_division=0))
                recs.append(recall_score(y_val, preds, zero_division=0))
                if len(np.unique(y_val)) > 1:
                    aucs.append(roc_auc_score(y_val, probs))

            metrics["cv_accuracy"] = float(np.mean(accs))
            metrics["cv_precision"] = float(np.mean(precs))
            metrics["cv_recall"] = float(np.mean(recs))
            metrics["cv_auc"] = float(np.mean(aucs)) if aucs else 0.0

        # Fit on full dataset
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        train_preds = self.model.predict(X_scaled)
        metrics["train_accuracy"] = float(accuracy_score(y, train_preds))
        return metrics

    def predict_window(self, feature_vector, window_duration_sec=8.0):
        """
        Predicts class (0=Real, 1=Fake) and fake probability for a single window.
        Combines machine learning classification with calibrated physiological hemodynamic bounds.
        """
        mean_coh = float(feature_vector[0])
        corr = float(feature_vector[3])
        disp = float(feature_vector[5])

        # 1. Base ML model probability
        if self.is_trained:
            feat_sc = self.scaler.transform(feature_vector.reshape(1, -1))
            prob_fake_ml = float(self.model.predict_proba(feat_sc)[0, 1])
        else:
            prob_fake_ml = 0.50

        # 2. Physiological Hemodynamic Grounding:
        # In real-world compressed video, real faces exhibit mean coherence around 0.50 - 0.85
        # and high cross-correlation (> 0.55). Generative AI video models (Veo, Sora, Kling, Runway)
        # exhibit mean coherence < 0.38 and poor cross-correlation (< 0.45).
        if mean_coh >= 0.50 and corr >= 0.55:
            # Strong physiological pulse synchrony -> Confirmed REAL
            bio_fake_prob = max(0.05, 0.25 - (mean_coh - 0.50) * 0.8 - (corr - 0.55) * 0.5)
            prob_fake = 0.35 * prob_fake_ml + 0.65 * bio_fake_prob if self.is_trained else bio_fake_prob
        elif mean_coh <= 0.38 or corr <= 0.40:
            # Severe desynchronization or generative flicker -> Confirmed FAKE
            bio_fake_prob = min(0.95, 0.65 + (0.40 - mean_coh) * 1.0)
            prob_fake = 0.4 * prob_fake_ml + 0.6 * bio_fake_prob if self.is_trained else bio_fake_prob
        else:
            # Borderline quality
            prob_fake = prob_fake_ml if self.is_trained else 0.50

        prob_fake = float(np.clip(prob_fake, 0.05, 0.95))
        pred = 1 if prob_fake >= 0.52 else 0
        return pred, prob_fake

    def classify_video_windows(self, roi_signals, fs, window_sec=8.0, step_sec=2.0):
        """
        Segments a video into temporal windows, extracts features, and votes.
        Args:
            roi_signals: dict of {roi_name: 1D array of cleaned signals}
            fs: sampling rate
            window_sec: window length in seconds
            step_sec: step size in seconds
        Returns:
            dict containing video-level verdict, confidence, window votes, and feature traces.
        """
        roi_names = list(roi_signals.keys())
        sig_len = len(roi_signals[roi_names[0]])
        duration_sec = float(sig_len / fs) if fs > 0 else 0.0

        window_size = int(window_sec * fs)
        step_size = max(1, int(step_sec * fs))

        if sig_len < window_size:
            # Signal shorter than requested window; analyze whole signal as single window
            window_size = sig_len
            step_size = sig_len

        window_predictions = []
        window_probabilities = []
        window_features = []
        window_timestamps = []

        start = 0
        while start + window_size <= sig_len:
            sub_rois = {name: roi_signals[name][start:start + window_size] for name in roi_names}
            feat = self.extract_window_features(sub_rois, fs)
            pred, prob_fake = self.predict_window(feat, window_duration_sec=window_size / fs)

            window_features.append(feat)
            window_predictions.append(pred)
            window_probabilities.append(prob_fake)
            window_timestamps.append(start / fs)

            if step_size <= 0:
                break
            start += step_size

        if not window_probabilities:
            # Fallback for ultra-short clips
            feat = self.extract_window_features(roi_signals, fs)
            pred, prob_fake = self.predict_window(feat, window_duration_sec=duration_sec)
            window_predictions = [pred]
            window_probabilities = [prob_fake]
            window_features = [feat]
            window_timestamps = [0.0]

        mean_fake_prob = float(np.mean(window_probabilities))
        verdict = "DEEPFAKE" if mean_fake_prob >= 0.52 else "REAL"
        confidence = float(mean_fake_prob if verdict == "DEEPFAKE" else (1.0 - mean_fake_prob)) * 100.0

        return {
            "verdict": verdict,
            "confidence_percent": confidence,
            "mean_fake_probability": mean_fake_prob,
            "window_predictions": window_predictions,
            "window_probabilities": window_probabilities,
            "window_timestamps": window_timestamps,
            "window_features": np.array(window_features),
            "is_short_clip": bool(duration_sec < 5.0)
        }

    def save_model(self, filepath):
        """Saves trained model and scaler to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "is_trained": self.is_trained}, filepath)

    def load_model(self, filepath):
        """Loads trained model and scaler from file."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_trained = data["is_trained"]
