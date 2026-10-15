"""
Spatial Coherence & Cross-Patch Consistency Module
===================================================
Analyzes the spatial consistency of rPPG pulse signals across multiple facial patches.
Implements:
- Pairwise Magnitude-Squared Coherence (MSC) via scipy.signal.coherence
- In-band cardiac coherence (0.7 - 3.0 Hz) and harmonic band (0.7 - 14.0 Hz)
- Normalized cross-correlation peak and phase delay
- Inter-patch frequency divergence (Delta BPM)
- Full Coherence Matrix computation
"""

import numpy as np
from scipy import signal
from .pulse_analyzer import PulseAnalyzer


class SpatialCoherenceAnalyzer:
    """Measures spatial synchrony and spectral coherence across facial ROI pulse signals."""

    @staticmethod
    def compute_pairwise_coherence(sig_a, sig_b, fs, lowcut=0.7, highcut=3.0):
        """
        Computes Magnitude-Squared Coherence (MSC) between two 1D signals within a target frequency band.
        MSC values lie in [0, 1].
        """
        n = len(sig_a)
        if n < 16 or fs <= 0:
            return 0.0, np.array([]), np.array([])

        nperseg = min(n, max(16, int(2.5 * fs)))
        freqs, coh = signal.coherence(sig_a, sig_b, fs=fs, nperseg=nperseg)

        in_band = (freqs >= lowcut) & (freqs <= highcut)
        if not np.any(in_band):
            return 0.0, freqs, coh

        mean_coh = float(np.mean(coh[in_band]))
        return mean_coh, freqs, coh

    @staticmethod
    def compute_cross_correlation(sig_a, sig_b):
        """
        Computes maximum normalized cross-correlation and optimal lag.
        Returns:
            max_corr: float [-1.0, 1.0]
            best_lag: int (in samples)
        """
        n = len(sig_a)
        if n < 4:
            return 0.0, 0

        norm_a = sig_a - np.mean(sig_a)
        norm_b = sig_b - np.mean(sig_b)
        std_a = np.std(norm_a)
        std_b = np.std(norm_b)

        if std_a == 0 or std_b == 0:
            return 0.0, 0

        norm_a /= std_a
        norm_b /= std_b

        corr = signal.correlate(norm_a, norm_b, mode='full') / n
        lags = signal.correlation_lags(n, n, mode='full')

        max_idx = np.argmax(corr)
        max_corr = float(corr[max_idx])
        best_lag = int(lags[max_idx])

        return max_corr, best_lag

    @classmethod
    def analyze_patch_set(cls, roi_signals, fs):
        """
        Analyzes a dictionary of clean pulse signals across multiple ROIs:
        e.g. {'forehead': sig1, 'left_cheek': sig2, 'right_cheek': sig3}

        Returns a detailed consistency dictionary:
            - 'coherence_matrix': NxN matrix of pairwise in-band coherence
            - 'mean_coherence': average off-diagonal coherence
            - 'min_coherence': lowest pairwise coherence
            - 'bpm_dispersion': standard deviation of estimated BPMs across ROIs
            - 'mean_cross_correlation': average maximum cross-correlation
            - 'mean_lag_abs': average absolute lag across pairs
            - 'roi_bpms': dict of {roi: bpm}
        """
        roi_names = sorted(list(roi_signals.keys()))
        n_rois = len(roi_names)

        if n_rois < 2:
            return {
                "coherence_matrix": np.ones((n_rois, n_rois)),
                "mean_coherence": 1.0,
                "min_coherence": 1.0,
                "bpm_dispersion": 0.0,
                "mean_cross_correlation": 1.0,
                "mean_lag_abs": 0.0,
                "roi_bpms": {r: 0.0 for r in roi_names},
                "roi_names": roi_names
            }

        coh_matrix = np.eye(n_rois)
        pairwise_coherences = []
        pairwise_corrs = []
        pairwise_lags = []
        roi_bpms = {}

        # 1. Estimate BPM per ROI
        for r_name in roi_names:
            sig = roi_signals[r_name]
            bpm_info = PulseAnalyzer.estimate_heart_rate(sig, fs)
            roi_bpms[r_name] = bpm_info["bpm"]

        # 2. Pairwise Coherence and Cross-Correlation
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                sig_i = roi_signals[roi_names[i]]
                sig_j = roi_signals[roi_names[j]]

                mean_coh, _, _ = cls.compute_pairwise_coherence(sig_i, sig_j, fs, lowcut=0.7, highcut=3.0)
                coh_matrix[i, j] = mean_coh
                coh_matrix[j, i] = mean_coh
                pairwise_coherences.append(mean_coh)

                corr, lag = cls.compute_cross_correlation(sig_i, sig_j)
                pairwise_corrs.append(corr)
                pairwise_lags.append(abs(lag))

        mean_coh = float(np.mean(pairwise_coherences)) if pairwise_coherences else 0.0
        min_coh = float(np.min(pairwise_coherences)) if pairwise_coherences else 0.0
        mean_corr = float(np.mean(pairwise_corrs)) if pairwise_corrs else 0.0
        mean_lag = float(np.mean(pairwise_lags)) if pairwise_lags else 0.0

        bpms_list = list(roi_bpms.values())
        bpm_std = float(np.std(bpms_list)) if len(bpms_list) > 1 else 0.0

        return {
            "coherence_matrix": coh_matrix,
            "mean_coherence": mean_coh,
            "min_coherence": min_coh,
            "bpm_dispersion": bpm_std,
            "mean_cross_correlation": mean_corr,
            "mean_lag_abs": mean_lag,
            "roi_bpms": roi_bpms,
            "roi_names": roi_names
        }
