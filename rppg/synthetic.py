"""
Synthetic Video & Signal Generator Module
=========================================
Generates physically grounded synthetic face videos and rPPG traces for testing,
unit tests, validation, and benchmarking without requiring 10 GB external downloads.
Simulates:
1. Authentic human face: synchronous Blood Volume Pulse (BVP) across forehead and cheeks,
   cardiac fundamental frequency + dicrotic notch + mild respiratory baseline drift.
2. Deepfake face: desynchronized patch frequencies, phase jitter, and high-frequency GAN flicker.
"""

import cv2
import numpy as np


class SyntheticPulseGenerator:
    """Generates synthetic pulse signals and test video clips."""

    @staticmethod
    def generate_bvp_signal(duration_sec, fps, bpm=72.0, add_noise=True, noise_level=0.08):
        """
        Synthesizes a realistic physiological BVP waveform with fundamental pulse
        and secondary dicrotic notch.
        """
        n_samples = int(duration_sec * fps)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        f_cardiac = bpm / 60.0

        # Fundamental systolic surge
        cardiac = 1.0 * np.sin(2.0 * np.pi * f_cardiac * t)
        # Secondary dicrotic wave (approx 2x frequency, phase shifted)
        dicrotic = 0.35 * np.sin(4.0 * np.pi * f_cardiac * t + np.pi / 3.0)
        # Respiration drift (approx 15 breaths/min = 0.25 Hz)
        respiration = 0.15 * np.sin(2.0 * np.pi * 0.25 * t)

        signal = cardiac + dicrotic + respiration

        if add_noise:
            noise = np.random.normal(0, noise_level, n_samples)
            signal += noise

        return t, signal

    @classmethod
    def generate_synthetic_video(cls, output_path, duration_sec=10.0, fps=30, is_fake=False,
                                 bpm=72.0, width=640, height=480):
        """
        Generates an MP4 video of an animated face with subtle physiological skin reflectance modulation.
        Args:
            output_path: str, destination .mp4 or .avi file path
            duration_sec: float, video length in seconds
            fps: int, frames per second
            is_fake: bool, True to inject deepfake desynchronization and flicker
            bpm: float, ground truth heart rate in beats per minute
            width: int, video width
            height: int, video height
        """
        n_frames = int(duration_sec * fps)
        t, bvp_real = cls.generate_bvp_signal(duration_sec, fps, bpm=bpm, noise_level=0.04)

        if is_fake:
            # Deepfake face patches: desynchronized or different frequencies + flicker
            _, bvp_forehead = cls.generate_bvp_signal(duration_sec, fps, bpm=bpm + 24.0, noise_level=0.15)
            _, bvp_left = cls.generate_bvp_signal(duration_sec, fps, bpm=bpm - 15.0, noise_level=0.15)
            # Right cheek has phase-inverted pulse
            bvp_right = -bvp_left + np.random.normal(0, 0.2, n_frames)
        else:
            # Authentic face: synchronous pulse across all facial patches
            bvp_forehead = bvp_real
            bvp_left = bvp_real + np.random.normal(0, 0.02, n_frames)
            bvp_right = bvp_real + np.random.normal(0, 0.02, n_frames)

        # Base skin tone in BGR (warm caucasian skin: approx B=140, G=175, R=225)
        base_skin_bgr = np.array([140.0, 175.0, 225.0], dtype=np.float32)

        # Blood surge increases absorption of green light (hence reflected green decreases slightly, approx 1-2%)
        pulse_amplitude = 2.5

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

        cx, cy = width // 2, height // 2
        face_rx, face_ry = width // 4, height // 3

        for i in range(n_frames):
            frame = np.full((height, width, 3), 40, dtype=np.uint8)

            # Draw face oval
            cv2.ellipse(frame, (cx, cy), (face_rx, face_ry), 0, 0, 360, (130, 165, 215), -1)

            # Physiological ROIs with modulated green reflectance
            # 1. Forehead patch
            mod_g_fh = -pulse_amplitude * bvp_forehead[i]
            col_fh = np.clip(base_skin_bgr + [0, mod_g_fh, 0], 0, 255).astype(np.uint8).tolist()
            cv2.rectangle(frame, (cx - 50, cy - 90), (cx + 50, cy - 40), col_fh, -1)

            # 2. Left cheek patch
            mod_g_lc = -pulse_amplitude * bvp_left[i]
            col_lc = np.clip(base_skin_bgr + [0, mod_g_lc, 0], 0, 255).astype(np.uint8).tolist()
            cv2.rectangle(frame, (cx - 85, cy + 10), (cx - 25, cy + 60), col_lc, -1)

            # 3. Right cheek patch
            mod_g_rc = -pulse_amplitude * bvp_right[i]
            col_rc = np.clip(base_skin_bgr + [0, mod_g_rc, 0], 0, 255).astype(np.uint8).tolist()
            cv2.rectangle(frame, (cx + 25, cy + 10), (cx + 85, cy + 60), col_rc, -1)

            # Draw facial features: eyes, nose, mouth
            cv2.ellipse(frame, (cx - 45, cy - 15), (15, 8), 0, 0, 360, (240, 240, 240), -1)
            cv2.circle(frame, (cx - 45, cy - 15), 5, (60, 40, 20), -1)
            cv2.ellipse(frame, (cx + 45, cy - 15), (15, 8), 0, 0, 360, (240, 240, 240), -1)
            cv2.circle(frame, (cx + 45, cy - 15), 5, (60, 40, 20), -1)

            # Nose
            pts_nose = np.array([[cx, cy - 5], [cx - 10, cy + 20], [cx + 10, cy + 20]], np.int32)
            cv2.polylines(frame, [pts_nose], True, (110, 140, 190), 2)

            # Mouth
            cv2.ellipse(frame, (cx, cy + 70), (30, 10), 0, 0, 180, (80, 80, 180), -1)

            # If fake, add sporadic GAN frame-jitter flicker
            if is_fake and np.random.rand() < 0.2:
                noise_flicker = np.random.randint(-4, 5, size=frame.shape, dtype=np.int16)
                frame = np.clip(frame.astype(np.int16) + noise_flicker, 0, 255).astype(np.uint8)

            out.write(frame)

        out.release()
        return output_path
