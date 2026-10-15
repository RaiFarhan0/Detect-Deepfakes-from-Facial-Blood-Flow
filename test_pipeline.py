"""
Automated Test Suite for rPPG Deepfake Detection System
======================================================
Tests every core algorithm and verifies end-to-end functionality:
1. Signal processing & Butterworth zero-phase filter
2. Pulse extraction (Green, CHROM, POS)
3. Frequency analysis & BPM recovery accuracy
4. Spatial patch coherence & cross-correlation
5. Deepfake classifier & window voting
"""

import unittest
import numpy as np
from rppg.rppg_extractor import RPPGExtractor
from rppg.signal_processor import SignalProcessor
from rppg.pulse_analyzer import PulseAnalyzer
from rppg.spatial_coherence import SpatialCoherenceAnalyzer
from rppg.detector import DeepfakePulseDetector
from rppg.synthetic import SyntheticPulseGenerator


class TestRPPGPipeline(unittest.TestCase):

    def setUp(self):
        self.fs = 30.0
        self.duration = 10.0
        self.target_bpm = 72.0

    def test_bvp_synthesis_and_bpm_estimation(self):
        """Verifies that an injected 72 BPM pulse is recovered within +/- 1.5 BPM."""
        t, bvp = SyntheticPulseGenerator.generate_bvp_signal(
            self.duration, self.fs, bpm=self.target_bpm, add_noise=False
        )
        filtered = SignalProcessor.clean_pulse_signal(bvp, fs=self.fs, lowcut=0.7, highcut=3.0)
        hr_info = PulseAnalyzer.estimate_heart_rate(filtered, fs=self.fs)

        detected_bpm = hr_info["bpm"]
        error = abs(detected_bpm - self.target_bpm)
        print(f"\n[Test] Target BPM: {self.target_bpm} | Detected: {detected_bpm:.2f} | Error: {error:.2f}")
        self.assertLess(error, 2.0, f"BPM estimation error too large: {error} BPM")
        self.assertGreater(hr_info["snr_db"], 5.0, "SNR of synthetic pure pulse should be high")

    def test_rppg_extraction_algorithms(self):
        """Tests that Green, CHROM, and POS execute cleanly on RGB matrices."""
        n_samples = int(self.duration * self.fs)
        # Synthetic RGB trace with slight modulation in Green channel
        t, pulse = SyntheticPulseGenerator.generate_bvp_signal(self.duration, self.fs, bpm=80.0)
        rgb_data = np.zeros((n_samples, 3))
        rgb_data[:, 0] = 200.0 + np.random.normal(0, 0.5, n_samples)
        rgb_data[:, 1] = 160.0 - 2.0 * pulse + np.random.normal(0, 0.5, n_samples)  # Green decreases with blood
        rgb_data[:, 2] = 130.0 + np.random.normal(0, 0.5, n_samples)

        sig_green = RPPGExtractor.extract(rgb_data, method="green")
        sig_chrom = RPPGExtractor.extract(rgb_data, method="chrom")
        sig_pos = RPPGExtractor.extract(rgb_data, method="pos")

        self.assertEqual(len(sig_green), n_samples)
        self.assertEqual(len(sig_chrom), n_samples)
        self.assertEqual(len(sig_pos), n_samples)

        # Check POS frequency recovery
        clean_pos = SignalProcessor.clean_pulse_signal(sig_pos, fs=self.fs)
        hr_pos = PulseAnalyzer.estimate_heart_rate(clean_pos, fs=self.fs)
        self.assertLess(abs(hr_pos["bpm"] - 80.0), 3.0)

    def test_spatial_coherence_real_vs_fake(self):
        """Verifies that synchronized patches yield high coherence, while desynchronized patches yield low coherence."""
        _, real_pulse = SyntheticPulseGenerator.generate_bvp_signal(self.duration, self.fs, bpm=72.0, noise_level=0.03)
        real_rois = {
            "forehead": real_pulse,
            "left_cheek": real_pulse + np.random.normal(0, 0.05, len(real_pulse)),
            "right_cheek": real_pulse + np.random.normal(0, 0.05, len(real_pulse))
        }
        stats_real = SpatialCoherenceAnalyzer.analyze_patch_set(real_rois, fs=self.fs)

        # Fake patches: completely uncorrelated or divergent frequencies
        _, fake_pulse_a = SyntheticPulseGenerator.generate_bvp_signal(self.duration, self.fs, bpm=60.0, noise_level=0.1)
        _, fake_pulse_b = SyntheticPulseGenerator.generate_bvp_signal(self.duration, self.fs, bpm=95.0, noise_level=0.1)
        fake_pulse_c = np.random.normal(0, 1.0, len(real_pulse))
        fake_rois = {
            "forehead": fake_pulse_a,
            "left_cheek": fake_pulse_b,
            "right_cheek": fake_pulse_c
        }
        stats_fake = SpatialCoherenceAnalyzer.analyze_patch_set(fake_rois, fs=self.fs)

        print(f"\n[Test] Synchronous Patches Coherence:   {stats_real['mean_coherence']:.3f}")
        print(f"[Test] Desynchronized Patches Coherence: {stats_fake['mean_coherence']:.3f}")

        self.assertGreater(stats_real["mean_coherence"], 0.70)
        self.assertLess(stats_fake["mean_coherence"], 0.50)
        self.assertGreater(stats_real["mean_cross_correlation"], stats_fake["mean_cross_correlation"])

    def test_detector_classification(self):
        """Verifies feature extraction and decision rules on real vs fake signals."""
        _, real_pulse = SyntheticPulseGenerator.generate_bvp_signal(self.duration, self.fs, bpm=72.0, noise_level=0.02)
        real_rois = {
            "forehead": SignalProcessor.clean_pulse_signal(real_pulse, fs=self.fs),
            "left_cheek": SignalProcessor.clean_pulse_signal(real_pulse + np.random.normal(0, 0.03, len(real_pulse)), fs=self.fs),
            "right_cheek": SignalProcessor.clean_pulse_signal(real_pulse + np.random.normal(0, 0.03, len(real_pulse)), fs=self.fs)
        }

        _, fake_a = SyntheticPulseGenerator.generate_bvp_signal(self.duration, self.fs, bpm=62.0, noise_level=0.15)
        _, fake_b = SyntheticPulseGenerator.generate_bvp_signal(self.duration, self.fs, bpm=98.0, noise_level=0.15)
        fake_rois = {
            "forehead": SignalProcessor.clean_pulse_signal(fake_a, fs=self.fs),
            "left_cheek": SignalProcessor.clean_pulse_signal(fake_b, fs=self.fs),
            "right_cheek": SignalProcessor.clean_pulse_signal(np.random.normal(0, 1.0, len(real_pulse)), fs=self.fs)
        }

        detector = DeepfakePulseDetector()
        res_real = detector.classify_video_windows(real_rois, fs=self.fs, window_sec=6.0, step_sec=2.0)
        res_fake = detector.classify_video_windows(fake_rois, fs=self.fs, window_sec=6.0, step_sec=2.0)

        print(f"\n[Test] Real verdict: {res_real['verdict']} (Confidence: {res_real['confidence_percent']:.1f}%)")
        print(f"[Test] Fake verdict: {res_fake['verdict']} (Confidence: {res_fake['confidence_percent']:.1f}%)")

        self.assertEqual(res_real["verdict"], "REAL")
        self.assertEqual(res_fake["verdict"], "DEEPFAKE")


if __name__ == "__main__":
    unittest.main()
