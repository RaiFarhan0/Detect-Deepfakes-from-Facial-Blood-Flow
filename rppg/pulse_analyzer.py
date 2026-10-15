"""
Pulse & Frequency Analyzer Module
=================================
Computes power spectral density via Fast Fourier Transform (FFT), identifies
the dominant cardiac frequency, estimates Heart Rate in Beats Per Minute (BPM),
and measures Signal-to-Noise Ratio (SNR) and spectral concentration.
"""

import numpy as np


class PulseAnalyzer:
    """Extracts cardiac frequency, BPM, and spectral power metrics using FFT."""

    @staticmethod
    def compute_fft(sig, fs, zero_padding_factor=4):
        """
        Computes the one-sided FFT amplitude spectrum with optional zero-padding
        for high-resolution peak localization.
        Args:
            sig: 1D np.ndarray.
            fs: float, sampling rate.
            zero_padding_factor: int, factor to increase FFT resolution.
        Returns:
            freqs: np.ndarray, frequency bins in Hz.
            power: np.ndarray, power spectrum (|FFT|^2).
        """
        n = len(sig)
        n_fft = n * zero_padding_factor
        # Apply Hanning window to reduce spectral leakage
        windowed = sig * np.hanning(n)
        fft_vals = np.fft.rfft(windowed, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        power = np.abs(fft_vals) ** 2
        return freqs, power

    @classmethod
    def estimate_heart_rate(cls, sig, fs, min_bpm=42.0, max_bpm=180.0, zero_padding_factor=4):
        """
        Estimates heart rate in BPM from the dominant frequency peak within the physiological band.
        Args:
            sig: 1D np.ndarray.
            fs: float, sampling rate.
            min_bpm: minimum realistic human heart rate.
            max_bpm: maximum realistic human heart rate.
            zero_padding_factor: interpolation factor.
        Returns:
            dict containing:
                'bpm': float, estimated beats per minute.
                'peak_freq_hz': float, dominant frequency in Hz.
                'snr_db': float, signal-to-noise ratio in dB.
                'spectral_entropy': float, normalized Shannon entropy of spectrum.
                'freqs': np.ndarray, frequency axis.
                'power': np.ndarray, power spectrum.
        """
        if len(sig) < 10 or fs <= 0:
            return {
                "bpm": 0.0,
                "peak_freq_hz": 0.0,
                "snr_db": 0.0,
                "spectral_entropy": 1.0,
                "freqs": np.array([]),
                "power": np.array([])
            }

        freqs, power = cls.compute_fft(sig, fs, zero_padding_factor=zero_padding_factor)

        min_hz = min_bpm / 60.0
        max_hz = max_bpm / 60.0

        in_band_mask = (freqs >= min_hz) & (freqs <= max_hz)
        if not np.any(in_band_mask):
            return {
                "bpm": 0.0,
                "peak_freq_hz": 0.0,
                "snr_db": 0.0,
                "spectral_entropy": 1.0,
                "freqs": freqs,
                "power": power
            }

        band_freqs = freqs[in_band_mask]
        band_power = power[in_band_mask]

        peak_idx = np.argmax(band_power)
        peak_freq = band_freqs[peak_idx]
        bpm = peak_freq * 60.0

        # Calculate SNR around peak (peak window +/- 0.15 Hz)
        peak_window_mask = (band_freqs >= peak_freq - 0.15) & (band_freqs <= peak_freq + 0.15)
        signal_energy = np.sum(band_power[peak_window_mask])
        noise_energy = np.sum(band_power[~peak_window_mask]) + 1e-8

        snr_db = 10.0 * np.log10(max(1e-8, signal_energy / noise_energy))

        # Normalized Spectral Entropy in cardiac band (lower = purer single tone, higher = white/flicker noise)
        p_norm = band_power / (np.sum(band_power) + 1e-12)
        # Avoid log(0)
        p_norm = p_norm[p_norm > 0]
        entropy = -np.sum(p_norm * np.log2(p_norm))
        max_entropy = np.log2(len(band_freqs)) if len(band_freqs) > 1 else 1.0
        norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 1.0

        return {
            "bpm": float(bpm),
            "peak_freq_hz": float(peak_freq),
            "snr_db": float(snr_db),
            "spectral_entropy": float(norm_entropy),
            "freqs": freqs,
            "power": power
        }
