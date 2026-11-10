"""Calcul de la Transformée de Fourier Discrète (DFT et FFT radix-2).

Ce module fournit une implémentation directe (O(N^2)) et une
implémentation FFT Cooley-Tukey radix-2 itérative pour signaux réels
et complexes, avec précalcul des facteurs twiddles et renversement de bits.
"""

from __future__ import annotations

import numpy as np

# Cache global des tables de renversement de bits 
_rev_cache: dict[int, np.ndarray] = {}

# Cache global des facteurs de rotation complexes pour la FFT radix-2
_twiddle_cache: dict[int, list[np.ndarray]] = {}

# Cache global des facteurs de rotation pour la décomposition FFT réelle
_rfft_twiddle_cache: dict[int, np.ndarray] = {}


def dft_naive(x: np.ndarray | list[complex] | list[float]) -> np.ndarray:
    """Calcul de la DFT avec la formule directe (double boucle for en O(N^2)).

    Formule mathématique :
        X[k] = somme_{n=0}^{N-1} x[n] * exp(-2j * pi * k * n / N)

    Args:
        x: Signal d'entrée (vecteur 1D réel ou complexe).

    Returns:
        np.ndarray: Spectre complexe de taille N.
    """
    arr = np.asarray(x, dtype=complex)
    n_pts = len(arr)
    spectrum = np.zeros(n_pts, dtype=complex)
    for k in range(n_pts):
        for n in range(n_pts):
            spectrum[k] += arr[n] * np.exp(-2j * np.pi * k * n / n_pts)
    return spectrum


def _bit_reversal(n_pts: int) -> np.ndarray:
    """Retourne la table de permutation par renversement de bits pour N."""
    if n_pts not in _rev_cache:
        bits = n_pts.bit_length() - 1
        idx = np.arange(n_pts)
        rev = np.zeros(n_pts, dtype=int)
        for b in range(bits):
            rev |= ((idx >> b) & 1) << (bits - 1 - b)
        _rev_cache[n_pts] = rev
    return _rev_cache[n_pts]


def _get_twiddles(n_pts: int) -> list[np.ndarray]:
    """Précalcule et met en cache les facteurs de rotation de chaque étage."""
    if n_pts not in _twiddle_cache:
        twiddles: list[np.ndarray] = []
        size = 2
        while size <= n_pts:
            half = size // 2
            twiddles.append(np.exp(-2j * np.pi * np.arange(half) / size))
            size *= 2
        _twiddle_cache[n_pts] = twiddles
    return _twiddle_cache[n_pts]


def _get_rfft_twiddles(n_pts: int) -> np.ndarray:
    """Précalcule les facteurs twiddles pour la décomposition réelle."""
    if n_pts not in _rfft_twiddle_cache:
        k = np.arange(n_pts // 2 + 1)
        _rfft_twiddle_cache[n_pts] = np.exp(-2j * np.pi * k / n_pts)
    return _rfft_twiddle_cache[n_pts]


def _fft_complex(x: np.ndarray) -> np.ndarray:
    """Algorithme FFT radix-2 itératif avec twiddles en cache."""
    n_pts = x.shape[0]
    result = x[_bit_reversal(n_pts)].copy()
    twiddles = _get_twiddles(n_pts)
    size = 2
    for twiddle in twiddles:
        half = size // 2
        result = result.reshape(-1, size)
        temp = result[:, half:] * twiddle
        result[:, half:] = result[:, :half] - temp
        result[:, :half] += temp
        size *= 2
    return result.reshape(-1)


def rfft_radix2(x: np.ndarray) -> np.ndarray:
    """FFT pour signal réel exploitant une FFT complexe de taille N/2.

    Réduit par deux le nombre d'opérations et la mémoire allouée en repliant
    deux échantillons réels consécutifs sous forme d'un échantillon complexe.

    Args:
        x: Signal 1D réel de longueur N (puissance de 2).

    Returns:
        np.ndarray: Spectre complexe unilatéral de taille N/2 + 1.
    """
    n_pts = len(x)
    if n_pts == 1:
        return np.asarray(x, dtype=complex)
    if n_pts == 0 or (n_pts & (n_pts - 1)) != 0:
        raise ValueError("N doit être une puissance de 2")

    half_n = n_pts // 2
    # Repliement de 2 échantillons réels par point complexe
    z = x[0::2] + 1j * x[1::2]
    big_z = _fft_complex(z)

    # Exploitation de la symétrie hermitienne
    big_z_rev = np.conj(np.roll(big_z[::-1], 1))
    w = _get_rfft_twiddles(n_pts)

    z_ext = np.empty(half_n + 1, dtype=complex)
    z_ext[:half_n] = big_z
    z_ext[half_n] = big_z[0]

    z_rev_ext = np.empty(half_n + 1, dtype=complex)
    z_rev_ext[:half_n] = big_z_rev
    z_rev_ext[half_n] = big_z_rev[0]

    return 0.5 * (z_ext + z_rev_ext) - 0.5j * w * (z_ext - z_rev_ext)


def fft_radix2(x: np.ndarray | list[complex] | list[float]) -> np.ndarray:
    """Version rapide de la DFT (FFT) Cooley-Tukey radix-2 itérative.

    Gère les signaux réels (via rfft_radix2 et symétrie) et complexes.

    Args:
        x: Signal d'entrée (taille N puissance de 2).

    Returns:
        np.ndarray: Spectre complexe complet de taille N.
    """
    arr = np.asarray(x)
    n_pts = arr.shape[0]
    if n_pts == 0 or (n_pts & (n_pts - 1)) != 0:
        raise ValueError("N doit être une puissance de 2")
    if n_pts == 1:
        return arr.astype(complex)

    # Détection de signal réel pour accélération
    if np.isrealobj(arr) or not np.iscomplexobj(arr) or np.all(np.isreal(arr)):
        x_real = arr.real if np.iscomplexobj(arr) else arr
        spectrum_half = rfft_radix2(x_real)
        full_spectrum = np.empty(n_pts, dtype=complex)
        half = n_pts // 2
        full_spectrum[:half + 1] = spectrum_half
        full_spectrum[half + 1:] = np.conj(spectrum_half[1:half][::-1])
        return full_spectrum

    return _fft_complex(arr.astype(complex))


def magnitude_spectrum(
    x: np.ndarray | list[complex] | list[float],
) -> np.ndarray:
    """Calcule le spectre d'amplitude (première moitié des fréquences)."""
    arr = np.asarray(x)
    n_pts = len(arr)
    if np.isrealobj(arr) or not np.iscomplexobj(arr) or np.all(np.isreal(arr)):
        x_real = arr.real if np.iscomplexobj(arr) else arr
        x_rfft = rfft_radix2(x_real)
        return np.abs(x_rfft[:n_pts // 2])
    x_full = fft_radix2(arr)
    return np.abs(x_full[:n_pts // 2])
