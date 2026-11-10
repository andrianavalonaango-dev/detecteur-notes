"""Calcul de la Transformée de Fourier Discrète (DFT)."""
import numpy as np


def dft_naive(x):
    """Calcul de la DFT avec la formule directe (deux boucles for)."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)
    # Formule directe de la DFT vue en cours :
    # X[k] = somme( x[n] * exp(-2j * pi * k * n / N) )
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


_rev_cache = {}


def _bit_reversal(N):
    # Réorganisation des indices pour l'algorithme FFT
    if N not in _rev_cache:
        bits = N.bit_length() - 1
        idx = np.arange(N)
        rev = np.zeros(N, dtype=int)
        for b in range(bits):
            rev |= ((idx >> b) & 1) << (bits - 1 - b)
        _rev_cache[N] = rev
    return _rev_cache[N]


def fft_radix2(x):
    """Version rapide de la DFT (FFT) pour le traitement en temps réel."""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    # N doit obligatoirement être une puissance de 2
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError("N doit être une puissance de 2")
    # On réordonne les échantillons au départ
    X = x[_bit_reversal(N)].copy()
    size = 2
    while size <= N:
        half = size // 2
        # Coefficients exponentiels de rotation
        twiddle = np.exp(-2j * np.pi * np.arange(half) / size)
        X = X.reshape(-1, size)
        even = X[:, :half].copy()
        odd = X[:, half:] * twiddle
        X[:, :half] = even + odd
        X[:, half:] = even - odd
        X = X.reshape(-1)
        size *= 2
    return X


def magnitude_spectrum(x):
    """Calcule le spectre d'amplitude (première moitié des fréquences)."""
    X = fft_radix2(x)
    return np.abs(X[: len(x) // 2])
