"""Détection de la fréquence fondamentale et analyse audio spectrale."""

from __future__ import annotations

import numpy as np

from dft import rfft_radix2
from notes import freq_to_note

# Paramètres standards du traitement du signal
SAMPLE_RATE: int = 44100        # Fréquence d'échantillonnage (44.1 kHz)
FRAME_SIZE: int = 8192          # Taille de trame temporelle (puissance de 2)
FMIN: float = 60.0              # Fréquence minimale d'intérêt (environ Si1)
FMAX: float = 1500.0            # Fréquence maximale d'intérêt (environ Sol6)
RMS_THRESHOLD: float = 0.01     # Seuil RMS pour détecter le silence

# Caches pour éviter les allocations et calculs répétitifs
_window_cache: dict[int, np.ndarray] = {}
_freqs_cache: dict[tuple[float, int], np.ndarray] = {}


def get_hanning_window(n_pts: int) -> np.ndarray:
    """Retourne une fenêtre de Hann précalculée pour la taille demandée."""
    if n_pts not in _window_cache:
        _window_cache[n_pts] = np.hanning(n_pts)
    return _window_cache[n_pts]


def spectrum(
    frame: np.ndarray | list[float],
    fs: float = float(SAMPLE_RATE),
    pad_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule le spectre d'amplitude avec fenêtrage et zero-padding.

    Le zero-padding d'un facteur 2 double la densité fréquentielle de la FFT,
    ce qui améliore considérablement la précision d'estimation des notes graves
    (ex. Mi2 = 82,41 Hz).

    Args:
        frame: Signal audio temporel.
        fs: Fréquence d'échantillonnage en Hz.
        pad_factor: Facteur d'extension par zéros (puissance de 2, défaut 2).

    Returns:
        tuple[np.ndarray, np.ndarray]: (vecteur des fréquences, amplitudes).
    """
    arr = np.asarray(frame, dtype=float)
    n_pts = len(arr)
    n_fft = n_pts * pad_factor

    # Application de la fenêtre de Hann et zero-padding
    windowed = np.zeros(n_fft, dtype=float)
    windowed[:n_pts] = arr * get_hanning_window(n_pts)

    # FFT optimisée pour signal réel (O(N log N))
    x_rfft = rfft_radix2(windowed)
    half_fft = n_fft // 2

    # Normalisation par la longueur initiale du signal utile
    mag = np.abs(x_rfft[:half_fft]) * (2.0 / n_pts)

    cache_key = (float(fs), n_fft)
    if cache_key not in _freqs_cache:
        _freqs_cache[cache_key] = (
            np.arange(half_fft, dtype=float) * (fs / n_fft)
        )
    freqs = _freqs_cache[cache_key]

    return freqs, mag


def hps_peak(
    mag: np.ndarray, freqs: np.ndarray, n_harm: int = 4
) -> int:
    """Trouve la fondamentale par Harmonic Product Spectrum (HPS) optimisé.

    Restreint le calcul à la bande utile [FMIN, FMAX] et à ses harmoniques
    pour minimiser les opérations et le temps de calcul.

    Args:
        mag: Spectre d'amplitude unilatéral.
        freqs: Vecteur de fréquences correspondant aux bins de mag.
        n_harm: Nombre d'harmoniques prises en compte (défaut 4).

    Returns:
        int: Indice du bin spectral correspondant à la fondamentale estimée.
    """
    lo = int(np.searchsorted(freqs, FMIN))
    hi = min(int(np.searchsorted(freqs, FMAX)), (len(mag) - 1) // n_harm)
    if hi <= lo:
        return lo

    # Plafond de dynamique pour éviter de multiplier par zéro
    max_val = float(np.max(mag))
    m = mag + (1e-3 * max_val)

    span = hi - lo
    hps = m[lo:hi].copy()
    for h in range(2, n_harm + 1):
        start = lo * h
        stop = start + span * h
        hps *= m[start:stop:h]

    # Élimination des bruits sans énergie spectrale significative
    hps[mag[lo:hi] < (0.05 * max_val)] = 0.0

    k = lo + int(np.argmax(hps))

    # Vérification sous-harmonique (correction d'octave basse)
    half_k = k // 2
    if half_k >= lo and (half_k * n_harm) < len(mag):
        idx_sub = half_k - lo
        idx_main = k - lo
        is_sub_valid = (
            hps[idx_sub] > (0.35 * hps[idx_main])
            and mag[half_k] > (0.1 * max_val)
        )
        if is_sub_valid:
            k = half_k

    return k


def parabolic_interp(mag: np.ndarray, k: int) -> float:
    """Affine l'indice du pic spectral par interpolation parabolique du log.

    Ajuste une parabole sur les 3 points du logarithme d'amplitude entourant
    le maximum pour obtenir une précision sous le demi-cent.

    Args:
        mag: Spectre d'amplitude.
        k: Indice du bin maximal entier.

    Returns:
        float: Indice affiné du sommet parabolique continu.
    """
    if k <= 0 or k >= len(mag) - 1:
        return float(k)

    a = float(np.log(mag[k - 1] + 1e-12))
    b = float(np.log(mag[k] + 1e-12))
    c = float(np.log(mag[k + 1] + 1e-12))

    denom = a - (2.0 * b) + c
    if denom == 0.0:
        return float(k)

    delta = 0.5 * (a - c) / denom
    return (k + delta) if abs(delta) <= 1.0 else float(k)


class PitchTracker:
    """Suivi temporel robuste du pitch pour flux audio continu.

    Fournit :
    - Lissage temporel par médiane glissante (5 estimations).
    - Suppression des sauts d'octave transitoires.
    - Seuil de silence adaptatif avec trigger de Schmitt.
    - Hystérésis d'affichage évitant le clignotement à la frontière de note.
    """

    def __init__(
        self,
        a4: float = 440.0,
        history_len: int = 5,
        base_threshold: float = RMS_THRESHOLD,
        hysteresis_cents: float = 6.0,
    ) -> None:
        self.a4 = a4
        self.history_len = history_len
        self.base_threshold = base_threshold
        self.hysteresis_cents = hysteresis_cents
        self.noise_floor = base_threshold * 0.5
        self.f0_history: list[float] = []
        self.octave_jump_count: int = 0
        self.active_note: str | None = None
        self.active_octave: int | None = None
        self.active_target: float | None = None
        self.silence_count: int = 0
        self.was_voiced: bool = False

    def reset(self) -> None:
        """Réinitialise l'historique et les états du tracker."""
        self.f0_history.clear()
        self.octave_jump_count = 0
        self.active_note = None
        self.active_octave = None
        self.active_target = None
        self.silence_count = 0
        self.was_voiced = False

    def process(self, frame_result: dict, rms: float) -> dict:
        """Applique les filtres temporels sur l'analyse instantanée."""
        adaptive_thresh = max(0.003, min(0.05, 2.5 * self.noise_floor))
        thresh_active = (
            adaptive_thresh * 0.8 if self.was_voiced else adaptive_thresh * 1.2
        )

        if rms < thresh_active:
            # Silence : mise à jour lente du plancher de bruit de fond
            self.noise_floor = (0.95 * self.noise_floor) + (0.05 * rms)
            self.silence_count += 1
            if self.silence_count >= 3:
                self.f0_history.clear()
                self.active_note = None
                self.active_octave = None
                self.active_target = None
                self.octave_jump_count = 0
            self.was_voiced = False
            res = frame_result.copy()
            res.update(
                voiced=False,
                f0=None,
                note=None,
                octave=None,
                cents=None,
                target=None,
            )
            return res

        self.was_voiced = True
        self.silence_count = 0
        raw_f0 = frame_result.get("f0")
        if raw_f0 is None or raw_f0 <= 0.0:
            return frame_result

        # Filtrage des sauts d'octave transitoires
        f0_candidate = raw_f0
        if self.f0_history:
            prev_median = float(np.median(self.f0_history))
            ratio = raw_f0 / prev_median
            is_octave_up = abs(ratio - 2.0) < 0.07
            is_octave_down = abs(ratio - 0.5) < 0.035
            if is_octave_up or is_octave_down:
                if self.octave_jump_count < 2:
                    f0_candidate = (
                        (raw_f0 / 2.0) if is_octave_up else (raw_f0 * 2.0)
                    )
                    self.octave_jump_count += 1
                else:
                    self.f0_history.clear()
                    self.octave_jump_count = 0
            else:
                self.octave_jump_count = 0

        # Lissage médian sur les 5 dernières trames
        self.f0_history.append(f0_candidate)
        if len(self.f0_history) > self.history_len:
            self.f0_history.pop(0)
        f0_smooth = float(np.median(self.f0_history))

        # Hystérésis d'affichage pour éviter l'oscillation Do / Do# à 50 cents
        raw_name, raw_oct, raw_cents, raw_tgt = freq_to_note(
            f0_smooth, self.a4
        )
        if self.active_target is not None and self.active_note is not None:
            c_active = 1200.0 * np.log2(f0_smooth / self.active_target)
            if abs(c_active) <= (50.0 + self.hysteresis_cents):
                n_name = self.active_note
                n_oct = self.active_octave
                n_cents = c_active
                n_tgt = self.active_target
            else:
                self.active_note = raw_name
                self.active_octave = raw_oct
                self.active_target = raw_tgt
                n_name, n_oct, n_cents, n_tgt = (
                    raw_name, raw_oct, raw_cents, raw_tgt
                )
        else:
            self.active_note = raw_name
            self.active_octave = raw_oct
            self.active_target = raw_tgt
            n_name, n_oct, n_cents, n_tgt = (
                raw_name, raw_oct, raw_cents, raw_tgt
            )

        res = frame_result.copy()
        res.update(
            voiced=True,
            f0=f0_smooth,
            note=n_name,
            octave=n_oct,
            cents=n_cents,
            target=n_tgt,
        )
        return res


def analyse_frame(
    frame: np.ndarray | list[float],
    fs: float = float(SAMPLE_RATE),
    a4: float = 440.0,
    rms_threshold: float = RMS_THRESHOLD,
    tracker: PitchTracker | None = None,
) -> dict:
    """Analyse un bloc audio et retourne les informations sur la note détectée.

    Args:
        frame: Vecteur des échantillons audio de la trame.
        fs: Fréquence d'échantillonnage en Hz.
        a4: Fréquence de référence du La4.
        rms_threshold: Seuil d'énergie RMS pour détecter le silence.
        tracker: Instance optionnelle de PitchTracker pour lissage temporel.

    Returns:
        dict: Dictionnaire contenant freqs, mag, rms, voiced, f0, note, octave,
              cents, target.
    """
    arr = np.asarray(frame, dtype=float)
    # Suppression de la composante continue (offset DC)
    centered = arr - float(np.mean(arr))
    rms = float(np.sqrt(np.mean(centered ** 2)))

    freqs, mag = spectrum(centered, fs)
    result = {
        "freqs": freqs,
        "mag": mag,
        "rms": rms,
        "voiced": False,
        "f0": None,
        "note": None,
        "octave": None,
        "cents": None,
        "target": None,
    }

    if tracker is not None:
        if rms < (rms_threshold * 0.1):
            return tracker.process(result, rms)
        k = hps_peak(mag, freqs)
        k_exact = parabolic_interp(mag, k)
        n_fft = len(mag) * 2
        f0 = k_exact * fs / n_fft
        name, octave, cents, target = freq_to_note(f0, a4)
        result.update(
            voiced=True,
            f0=f0,
            note=name,
            octave=octave,
            cents=cents,
            target=target,
        )
        return tracker.process(result, rms)

    # Analyse instantanée autonome (pour tests unitaires et analyse isolée)
    if rms < rms_threshold:
        return result

    k = hps_peak(mag, freqs)
    k_exact = parabolic_interp(mag, k)
    n_fft = len(mag) * 2
    f0 = k_exact * fs / n_fft
    name, octave, cents, target = freq_to_note(f0, a4)
    result.update(
        voiced=True,
        f0=f0,
        note=name,
        octave=octave,
        cents=cents,
        target=target,
    )
    return result
