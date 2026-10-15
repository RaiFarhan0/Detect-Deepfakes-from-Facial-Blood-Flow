"""
Visualizer Module
=================
Generates diagnostic figures and video overlays:
- Temporal BVP wave plots (raw vs filtered)
- FFT power spectral density with heart rate peak indicators
- Spatial Coherence Matrix heatmap
- Real vs Deepfake verdict banner and metric radar
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PulseVisualizer:
    """Visualizes rPPG waveforms, frequency spectra, and spatial coherence matrices."""

    @staticmethod
    def plot_diagnostic(roi_clean_signals, fs, hr_info, coh_stats, verdict_info, save_path="diagnostic.png"):
        """
        Creates a 4-panel diagnostic plot:
        1. Temporal filtered BVP waveforms for each facial patch
        2. FFT Power Spectral Density and dominant BPM peak
        3. Pairwise Spatial Coherence Matrix
        4. Verdict summary with metric bars
        """
        fig = plt.figure(figsize=(14, 10))
        plt.style.use('fast')

        # 1. Temporal Waveforms
        ax1 = fig.add_subplot(2, 2, 1)
        roi_names = sorted(list(roi_clean_signals.keys()))
        t = np.linspace(0, len(roi_clean_signals[roi_names[0]]) / fs, len(roi_clean_signals[roi_names[0]]))
        # Show first 8 seconds for visual clarity
        t_mask = t <= 8.0

        colors = {'forehead': '#2ca02c', 'left_cheek': '#1f77b4', 'right_cheek': '#ff7f0e', 'nose': '#9467bd', 'chin': '#8c564b'}
        for name in roi_names:
            c = colors.get(name, '#333333')
            ax1.plot(t[t_mask], roi_clean_signals[name][t_mask], label=f"{name.replace('_', ' ').title()}", color=c, lw=1.8, alpha=0.85)

        ax1.set_title("1. Filtered rPPG Pulse Traces (Spatial Patches)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Normalized BVP Amplitude")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. FFT Power Spectrum
        ax2 = fig.add_subplot(2, 2, 2)
        freqs = hr_info.get("freqs", np.array([]))
        power = hr_info.get("power", np.array([]))
        if len(freqs) > 0 and len(power) > 0:
            bpm_axis = freqs * 60.0
            mask_hr = (bpm_axis >= 40.0) & (bpm_axis <= 200.0)
            ax2.plot(bpm_axis[mask_hr], power[mask_hr], color='#d62728', lw=2.0)
            peak_bpm = hr_info.get("bpm", 0.0)
            ax2.axvline(peak_bpm, color='#2ca02c', linestyle='--', lw=1.5, label=f"Detected: {peak_bpm:.1f} BPM ({peak_bpm/60:.2f} Hz)")
            ax2.set_title("2. Cardiac Power Spectrum (FFT)", fontsize=12, fontweight='bold')
            ax2.set_xlabel("Frequency (Beats Per Minute)")
            ax2.set_ylabel("Spectral Power (|FFT|^2)")
            ax2.legend(loc="upper right", fontsize=9)
            ax2.grid(True, alpha=0.3)

        # 3. Spatial Coherence Heatmap
        ax3 = fig.add_subplot(2, 2, 3)
        coh_matrix = coh_stats.get("coherence_matrix", np.eye(len(roi_names)))
        im = ax3.imshow(coh_matrix, cmap='YlGnBu', vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label("Magnitude Squared Coherence")

        labels = [name.replace('_', '\n').title() for name in coh_stats.get("roi_names", roi_names)]
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, fontsize=10)
        ax3.set_yticklabels(labels, fontsize=10)

        # Annotate numbers in cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = coh_matrix[i, j]
                ax3.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.6 else "black", fontweight="bold")

        ax3.set_title(f"3. Inter-Patch Coherence Matrix (Mean = {coh_stats.get('mean_coherence', 0.0):.2f})", fontsize=12, fontweight='bold')

        # 4. Verdict and Metric Breakdown
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        verdict = verdict_info.get("verdict", "UNKNOWN")
        confidence = verdict_info.get("confidence_percent", 0.0)
        is_real = (verdict == "REAL")
        badge_color = "#2ca02c" if is_real else "#d62728"

        ax4.text(0.5, 0.88, f"VERDICT: {verdict}", ha="center", va="center",
                 fontsize=18, fontweight='heavy', color=badge_color,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor=badge_color, lw=2.5))

        ax4.text(0.5, 0.72, f"Confidence: {confidence:.1f}%  |  Estimated Pulse: {hr_info.get('bpm', 0.0):.1f} BPM",
                 ha="center", va="center", fontsize=12, fontweight='bold')

        # Feature Summary Metrics
        metrics_text = (
            f"- Mean Spatial Coherence:      {coh_stats.get('mean_coherence', 0.0):.3f}   (Real faces usually > 0.65)\n"
            f"- Minimum Patch Coherence:     {coh_stats.get('min_coherence', 0.0):.3f}\n"
            f"- Mean Cross-Correlation:       {coh_stats.get('mean_cross_correlation', 0.0):.3f}   (Real faces usually > 0.55)\n"
            f"- Mean Phase Lag (samples):     {coh_stats.get('mean_lag_abs', 0.0):.2f}    (Near 0 for synchronized blood flow)\n"
            f"- Inter-patch BPM Dispersion:   {coh_stats.get('bpm_dispersion', 0.0):.2f} BPM (Zero when all patches match)\n"
            f"- Cardiac Peak SNR:             {hr_info.get('snr_db', 0.0):.2f} dB   (Higher = cleaner physiological pulse)\n"
            f"- Normalized Spectral Entropy:  {hr_info.get('spectral_entropy', 1.0):.3f}   (Lower = concentrated cardiac peak)"
        )
        ax4.text(0.08, 0.28, metrics_text, family="monospace", fontsize=10, va="center")

        plt.tight_layout()
        plt.savefig(save_path, dpi=180)
        plt.close(fig)
        return save_path

    @staticmethod
    def annotate_frame(frame, rois, bpm=None, verdict=None, confidence=None):
        """
        Draws bounding boxes around active facial patches and a HUD on the video frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw ROI bounding boxes
        if rois:
            colors = {
                'forehead': (0, 220, 0),
                'left_cheek': (220, 150, 0),
                'right_cheek': (0, 150, 220),
                'nose': (180, 0, 180),
                'chin': (120, 120, 120)
            }
            for name, info in rois.items():
                bx, by, bw, bh = info["bbox"]
                col = colors.get(name, (0, 255, 0))
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), col, 2)
                cv2.putText(annotated, name.title(), (bx, max(15, by - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # Header HUD
        hud_h = 45
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # BPM text
        bpm_str = f"Pulse: {bpm:.1f} BPM" if bpm is not None and bpm > 0 else "Pulse: Calculating..."
        cv2.putText(annotated, bpm_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA)

        # Verdict text
        if verdict:
            is_real = (verdict == "REAL")
            v_col = (0, 230, 0) if is_real else (0, 0, 240)
            conf_str = f"({confidence:.0f}%)" if confidence else ""
            cv2.putText(annotated, f"{verdict} {conf_str}", (w - 220, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_col, 2, cv2.LINE_AA)

        return annotated
