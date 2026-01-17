import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import zscore
from typing import List, Optional
from .data_structures import SeismicFeatures, CMPGather
from .avo_classification import classify_avo


class FeatureExtractor:
    def __init__(self, cmp: CMPGather):
        self.cmp = cmp

    def extract(self, y: np.ndarray, angles: np.ndarray) -> SeismicFeatures:
        zero_crossing_rate = self._compute_zero_crossing_rate(y)
        dominant_frequency, bandwidth = self._compute_frequency_features(y)
        energy_envelope_mean, energy_decay_rate = self._compute_energy_features(y)
        dynamic_range_db = self._compute_dynamic_range(y)
        linear_trend_slope, curvature, trend_r_squared = self._compute_trend_features(y)
        outlier_indices, max_z_score = self._detect_outliers(y)
        phase_reversals = self._count_phase_reversals(y)
        avo_result = classify_avo(angles, y)
        periodicity_score, dominant_period = self._compute_periodicity(y)

        return SeismicFeatures(
            zero_crossing_rate=zero_crossing_rate,
            dominant_frequency=dominant_frequency,
            bandwidth=bandwidth,
            energy_envelope_mean=energy_envelope_mean,
            energy_decay_rate=energy_decay_rate,
            dynamic_range_db=dynamic_range_db,
            linear_trend_slope=linear_trend_slope,
            curvature=curvature,
            trend_r_squared=trend_r_squared,
            outlier_indices=outlier_indices,
            max_z_score=max_z_score,
            phase_reversals=phase_reversals,
            avo_type=avo_result.avo_type.value,
            intercept=avo_result.intercept_A,
            gradient=avo_result.gradient_B,
            intercept_gradient_ratio=avo_result.ratio_AB,
            periodicity_score=periodicity_score,
            dominant_period=dominant_period
        )

    def _compute_zero_crossing_rate(self, y: np.ndarray) -> float:
        sign_y = np.sign(y)
        sign_changes = np.sum(np.abs(np.diff(sign_y)) > 1)
        zcr = sign_changes / (len(y) - 1)
        return zcr

    def _compute_frequency_features(self, y: np.ndarray) -> tuple:
        n = len(y)
        y_fft = fft(y)
        freqs = fftfreq(n, d=self.cmp.dt)
        power = np.abs(y_fft)**2
        positive_freqs = freqs[:n//2]
        positive_power = power[:n//2]

        dominant_idx = np.argmax(positive_power)
        dominant_frequency = positive_freqs[dominant_idx]

        threshold = 0.5 * np.max(positive_power)
        bandwidth = positive_freqs[positive_power > threshold][-1] - positive_freqs[positive_power > threshold][0]

        return dominant_frequency, bandwidth

    def _compute_energy_features(self, y: np.ndarray) -> tuple:
        analytic_signal = signal.hilbert(y)
        envelope = np.abs(analytic_signal)
        energy_envelope_mean = np.mean(envelope)

        x = np.arange(len(y))
        log_envelope = np.log(envelope + 1e-10)
        coeffs = np.polyfit(x, log_envelope, 1)
        energy_decay_rate = -coeffs[0]

        return energy_envelope_mean, energy_decay_rate

    def _compute_dynamic_range(self, y: np.ndarray) -> float:
        max_amp = np.max(np.abs(y))
        min_amp = np.min(np.abs(y))
        if min_amp < 1e-10:
            return 20 * np.log10(max_amp)
        dynamic_range_db = 20 * np.log10(max_amp / min_amp)
        return dynamic_range_db

    def _compute_trend_features(self, y: np.ndarray) -> tuple:
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 1)
        linear_trend_slope = coeffs[0]

        d2y = np.diff(y, n=2)
        curvature = np.mean(np.abs(d2y))

        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        trend_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return linear_trend_slope, curvature, trend_r_squared

    def _detect_outliers(self, y: np.ndarray, threshold: float = 3.0) -> tuple:
        z_scores = np.abs(zscore(y))
        outlier_indices = np.where(z_scores > threshold)[0].tolist()
        max_z_score = np.max(z_scores) if len(z_scores) > 0 else 0
        return outlier_indices, max_z_score

    def _count_phase_reversals(self, y: np.ndarray) -> int:
        sign_changes = 0
        for i in range(len(y) - 1):
            if y[i] * y[i+1] < 0:
                sign_changes += 1
        return sign_changes

    def _compute_periodicity(self, y: np.ndarray) -> tuple:
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
        if len(peaks) == 0:
            return 0.0, None

        periodicity_score = np.max(autocorr[peaks + 1])
        dominant_period_idx = peaks[np.argmax(autocorr[peaks + 1])] + 1
        dominant_period = float(dominant_period_idx)

        return periodicity_score, dominant_period
