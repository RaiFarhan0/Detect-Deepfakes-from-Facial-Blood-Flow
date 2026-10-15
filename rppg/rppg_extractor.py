"""
rPPG Signal Extractor Module
============================
Transforms raw temporal RGB traces into physiological Blood Volume Pulse (BVP) signals.
Implements:
1. Green channel baseline (Verkruysse et al., 2008)
2. CHROM: Chrominance-based rPPG (de Haan & Jeanne, 2013)
3. POS: Plane-Orthogonal-to-Skin rPPG (Wang et al., 2017)
"""

import numpy as np


class RPPGExtractor:
    """Extracts BVP pulse signals from temporal RGB traces using Green, CHROM, or POS."""

    @staticmethod
    def extract_green(rgb_array):
        """
        Extracts rPPG signal using the normalized Green channel.
        Args:
            rgb_array: np.ndarray of shape (N, 3), columns [R, G, B].
        Returns:
            np.ndarray of shape (N,) containing raw BVP trace.
        """
        green = rgb_array[:, 1]
        mean_g = np.mean(green)
        if mean_g == 0:
            return np.zeros_like(green)
        return (green / mean_g) - 1.0

    @staticmethod
    def extract_chrom(rgb_array):
        """
        Extracts rPPG signal using the CHROM (Chrominance) algorithm.
        Standardizes R, G, B traces and projects onto orthogonal chrominance planes:
            Xs = 3*Rn - 2*Gn
            Ys = 1.5*Rn + Gn - 1.5*Bn
            S = Xs - alpha * Ys, where alpha = std(Xs) / std(Ys)
        """
        if len(rgb_array) < 4:
            return np.zeros(len(rgb_array))

        # Temporal normalization
        mean_rgb = np.mean(rgb_array, axis=0, keepdims=True)
        mean_rgb[mean_rgb == 0] = 1e-6
        norm_rgb = rgb_array / mean_rgb

        r = norm_rgb[:, 0]
        g = norm_rgb[:, 1]
        b = norm_rgb[:, 2]

        xs = 3.0 * r - 2.0 * g
        ys = 1.5 * r + g - 1.5 * b

        std_ys = np.std(ys)
        if std_ys == 0:
            return xs
        alpha = np.std(xs) / std_ys
        bvp = xs - alpha * ys
        return bvp

    @staticmethod
    def extract_pos(rgb_array, window_length=30):
        """
        Extracts rPPG signal using Plane-Orthogonal-to-Skin (POS).
        Projects normalized color vectors onto a plane orthogonal to the skin-tone vector.
        Args:
            rgb_array: np.ndarray of shape (N, 3)
            window_length: sliding window size in frames
        Returns:
            np.ndarray of shape (N,)
        """
        n = len(rgb_array)
        if n < 4:
            return np.zeros(n)

        window_length = min(window_length, n)
        h = np.zeros(n)

        # Sliding window projection
        for t in range(0, n - window_length + 1):
            sub_rgb = rgb_array[t:t + window_length]
            mean_c = np.mean(sub_rgb, axis=0, keepdims=True)
            mean_c[mean_c == 0] = 1e-6
            cn = sub_rgb / mean_c

            s1 = -cn[:, 1] + cn[:, 2]
            s2 = cn[:, 0] - 2.0 * cn[:, 1] + cn[:, 2]

            std_s2 = np.std(s2)
            alpha = np.std(s1) / (std_s2 if std_s2 > 0 else 1e-6)
            s = s1 + alpha * s2
            s_centered = s - np.mean(s)

            h[t:t + window_length] += s_centered

        # Pad remainder if any
        if n > window_length:
            h[:window_length] = h[:window_length] / np.linspace(1, window_length, window_length)
            h[-window_length:] = h[-window_length:] / np.linspace(window_length, 1, window_length)

        return h

    @classmethod
    def extract(cls, rgb_array, method="pos", window_length=30):
        """Extracts BVP signal using chosen method ('green', 'chrom', or 'pos')."""
        method = method.lower()
        if method == "green":
            return cls.extract_green(rgb_array)
        elif method == "chrom":
            return cls.extract_chrom(rgb_array)
        elif method == "pos":
            return cls.extract_pos(rgb_array, window_length=window_length)
        else:
            raise ValueError(f"Unknown rPPG extraction method: {method}")
