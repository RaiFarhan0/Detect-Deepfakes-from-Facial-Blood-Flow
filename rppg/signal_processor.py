"""
Signal Processor Module
=======================
Applies temporal detrending, Butterworth bandpass filtering, and normalization
to isolate physiological cardiac frequencies (0.7 Hz - 3.0 Hz, or 42 - 180 BPM)
and FakeCatcher extended harmonics (0.7 Hz - 14.0 Hz).
"""

import numpy as np
from scipy import signal


class SignalProcessor:
    """Preprocesses and filters raw BVP signals using Butterworth IIR filtering and detrending."""

    @staticmethod
    def detrend_signal(raw_signal):
        """Removes slow linear and low-frequency baseline drifts."""
        if len(raw_signal) < 2:
            return raw_signal
        return signal.detrend(raw_signal, type='linear')

    @staticmethod
    def bandpass_filter(raw_signal, fs, lowcut=0.7, highcut=3.0, order=3):
        """
        Applies zero-phase Butterworth bandpass filter.
        Args:
            raw_signal: np.ndarray, 1D signal.
            fs: float, sampling rate (video FPS).
            lowcut: float, lower cutoff frequency in Hz (default 0.7 Hz = 42 BPM).
            highcut: float, upper cutoff frequency in Hz (default 3.0 Hz = 180 BPM).
            order: int, filter order.
        Returns:
            np.ndarray: filtered zero-phase signal.
        """
        n = len(raw_signal)
        if n < 10:
            return raw_signal

        # Nyquist frequency safeguard
        nyquist = 0.5 * fs
        if highcut >= nyquist:
            highcut = nyquist * 0.95
        if lowcut >= highcut:
            lowcut = highcut * 0.2

        try:
            sos = signal.butter(
                order,
                [lowcut, highcut],
                btype='bandpass',
                fs=fs,
                output='sos'
            )
            # Use zero-phase forward-backward filtering to avoid phase distortion
            filtered = signal.sosfiltfilt(sos, raw_signal)
            return filtered
        except Exception:
            # Fallback if signal is too short for chosen order
            return raw_signal - np.mean(raw_signal)

    @staticmethod
    def normalize_signal(sig):
        """Normalizes signal to zero mean and unit variance."""
        std = np.std(sig)
        if std == 0:
            return sig - np.mean(sig)
        return (sig - np.mean(sig)) / std

    @classmethod
    def clean_pulse_signal(cls, raw_signal, fs, lowcut=0.7, highcut=3.0, order=3):
        """Full pipeline: detrend -> bandpass filter -> standardize."""
        detrended = cls.detrend_signal(raw_signal)
        filtered = cls.bandpass_filter(detrended, fs=fs, lowcut=lowcut, highcut=highcut, order=order)
        standardized = cls.normalize_signal(filtered)
        return standardized
