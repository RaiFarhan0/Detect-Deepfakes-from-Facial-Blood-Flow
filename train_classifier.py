"""
Model Training & GroupKFold Cross-Validation Script
===================================================
Trains the SVM classifier on realistic multi-tier rPPG data:
- Authentic data includes studio quality, webcam quality, and compressed mobile video
- Deepfake data simulates generative diffusion/GAN flicker and patch desynchronization
- Enforces GroupKFold cross-validation by person/subject identity (zero identity leakage)
"""

import os
import numpy as np
from rppg.detector import DeepfakePulseDetector
from rppg.synthetic import SyntheticPulseGenerator
from rppg.signal_processor import SignalProcessor


def build_realistic_training_dataset(num_subjects=16, clips_per_subject=6, duration_sec=10.0, fs=30.0):
    """
    Builds a robust training dataset representing real-world video quality variations:
    authentic clips range from studio clean to compressed webcam, preventing false positives.
    """
    X = []
    y = []
    groups = []

    print(f"Generating realistic training dataset ({num_subjects} subjects, {clips_per_subject} clips each)...")

    for subject_id in range(num_subjects):
        subject_base_bpm = 58.0 + (subject_id * 3.7) % 40.0

        for clip_id in range(clips_per_subject):
            is_fake = (clip_id % 2 == 1)
            clip_bpm = subject_base_bpm + np.random.uniform(-3.0, 3.0)

            if not is_fake:
                # Real face: vary noise level to simulate clean studio (0.03), webcam (0.08), compressed (0.14)
                noise_lvl = np.random.choice([0.03, 0.07, 0.12, 0.16])
                _, pulse = SyntheticPulseGenerator.generate_bvp_signal(
                    duration_sec, fs, bpm=clip_bpm, noise_level=noise_lvl
                )

                # Realistic capillary vascular delay between forehead and cheeks (0 to 2 frames)
                delay_l = np.random.choice([0, 1])
                delay_r = np.random.choice([0, 1])

                pulse_l = np.roll(pulse, delay_l) + np.random.normal(0, noise_lvl * 0.5, len(pulse))
                pulse_r = np.roll(pulse, delay_r) + np.random.normal(0, noise_lvl * 0.5, len(pulse))

                roi_signals = {
                    "forehead": SignalProcessor.clean_pulse_signal(pulse, fs=fs),
                    "left_cheek": SignalProcessor.clean_pulse_signal(pulse_l, fs=fs),
                    "right_cheek": SignalProcessor.clean_pulse_signal(pulse_r, fs=fs)
                }
                label = 0  # REAL
            else:
                # Deepfake face:
                # Type A: Patch desynchronization (different heart rates in different face parts)
                # Type B: AI video generative noise (Veo, Sora, Runway, StyleGAN) with chaotic phase
                fake_type = np.random.choice(["desync", "generative_noise"])
                if fake_type == "desync":
                    _, pulse_fh = SyntheticPulseGenerator.generate_bvp_signal(
                        duration_sec, fs, bpm=clip_bpm + np.random.choice([-24.0, 22.0]), noise_level=0.15
                    )
                    _, pulse_lc = SyntheticPulseGenerator.generate_bvp_signal(
                        duration_sec, fs, bpm=clip_bpm + np.random.choice([-18.0, 16.0]), noise_level=0.15
                    )
                    pulse_rc = -pulse_lc + np.random.normal(0, 0.25, len(pulse_fh))
                else:
                    # Generative diffusion / GAN noise: low cross-patch correlation (<0.35)
                    n_pts = int(duration_sec * fs)
                    t_pts = np.linspace(0, duration_sec, n_pts, endpoint=False)
                    # Random flickering signals
                    pulse_fh = np.sin(2.0 * np.pi * 0.8 * t_pts) + np.random.normal(0, 0.5, n_pts)
                    pulse_lc = np.sin(2.0 * np.pi * 2.2 * t_pts + 1.2) + np.random.normal(0, 0.5, n_pts)
                    pulse_rc = np.random.normal(0, 0.8, n_pts)

                roi_signals = {
                    "forehead": SignalProcessor.clean_pulse_signal(pulse_fh, fs=fs),
                    "left_cheek": SignalProcessor.clean_pulse_signal(pulse_lc, fs=fs),
                    "right_cheek": SignalProcessor.clean_pulse_signal(pulse_rc, fs=fs)
                }
                label = 1  # DEEPFAKE

            # Extract sliding window features (5-second windows, 2.5-second step)
            win_size = int(5.0 * fs)
            step_size = int(2.5 * fs)
            sig_len = len(roi_signals["forehead"])

            for start in range(0, sig_len - win_size + 1, step_size):
                sub_rois = {k: roi_signals[k][start:start + win_size] for k in roi_signals}
                feat = DeepfakePulseDetector.extract_window_features(sub_rois, fs=fs)
                X.append(feat)
                y.append(label)
                groups.append(subject_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups, dtype=np.int32)

    print(f"Generated {len(X)} feature windows across {num_subjects} subjects.")
    print(f"Class distribution: Real = {np.sum(y == 0)}, Deepfake = {np.sum(y == 1)}")
    return X, y, groups


def main():
    print("=" * 60)
    print("  TRAINING REALISTIC RPPG DEEPFAKE DETECTOR (ROBUST REAL-WORLD)")
    print("=" * 60)

    X, y, groups = build_realistic_training_dataset(num_subjects=16, clips_per_subject=6)

    detector = DeepfakePulseDetector(classifier_type="svm")
    print("\nRunning 5-Fold GroupKFold Cross-Validation (split by Subject ID)...")
    metrics = detector.train(X, y, groups=groups, n_splits=5)

    print("\n" + "-" * 60)
    print("  CROSS-VALIDATION RESULTS (PERSON-HELD-OUT):")
    print(f"    - Mean CV Accuracy:  {metrics['cv_accuracy'] * 100:.1f}%")
    print(f"    - Mean CV Precision: {metrics['cv_precision'] * 100:.1f}%")
    print(f"    - Mean CV Recall:    {metrics['cv_recall'] * 100:.1f}%")
    print(f"    - Mean CV ROC-AUC:   {metrics['cv_auc']:.3f}")
    print(f"    - Full Train Acc:    {metrics['train_accuracy'] * 100:.1f}%")
    print("-" * 60)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rppg_model.joblib")
    detector.save_model(model_path)
    print(f"\nModel weights saved successfully to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
