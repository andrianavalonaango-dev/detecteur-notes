"""Détection de la fréquence fondamentale et analyse audio."""
import numpy as np
from dft import fft_radix2
from notes import freq_to_note

# Paramètres généraux
SAMPLE_RATE = 44100        # Fréquence d'échantillonnage (44.1 kHz standard)
FRAME_SIZE = 8192          # Nombre d'échantillons par bloc analysé
FMIN, FMAX = 60.0, 1500.0  # Plage de fréquences musicales à analyser
RMS_THRESHOLD = 0.01       # Seuil d'énergie sous lequel on considère que c'est du silence


def spectrum(frame, fs):
    """Calcule le spectre d'amplitude d'un bloc audio."""
    N = len(frame)
    # On applique une fenêtre de Hann pour lisser les bords du signal
    windowed = frame * np.hanning(N)
    # Calcul de la FFT
    X = fft_radix2(windowed)
    # On garde la moitié utile et on normalise l'amplitude
    mag = np.abs(X[: N // 2]) * 2 / N
    # Fréquences associées à chaque point
    freqs = np.arange(N // 2) * fs / N
    return freqs, mag


def hps_peak(mag, freqs, n_harm=4):
    """Trouve la fondamentale grâce à la méthode HPS (Harmonic Product Spectrum)."""
    L = len(mag) // n_harm
    # On ajoute une petite constante pour éviter de multiplier par 0
    m = mag + 1e-3 * mag.max()
    hps = m[:L].copy()
    # On multiplie avec les sous-échantillonnages pour faire ressortir la fréquence de base
    for h in range(2, n_harm + 1):
        hps *= m[::h][:L]
    # On met à zéro si le signal de base est trop faible
    hps[mag[:L] < 0.05 * mag.max()] = 0
    # On limite la recherche entre FMIN et FMAX
    lo = int(np.searchsorted(freqs, FMIN))
    hi = min(int(np.searchsorted(freqs, FMAX)), L - 1)
    return lo + int(np.argmax(hps[lo:hi]))


def parabolic_interp(mag, k):
    """Affine l'indice du pic maximal avec une interpolation parabolique."""
    if k <= 0 or k >= len(mag) - 1:
        return float(k)
    # Logarithme des 3 points autour du pic
    a, b, c = np.log(mag[k - 1] + 1e-12), np.log(mag[k] + 1e-12), np.log(mag[k + 1] + 1e-12)
    denom = a - 2 * b + c
    if denom == 0:
        return float(k)
    # Décalage estimé du sommet de la parabole
    delta = 0.5 * (a - c) / denom
    return k + delta if abs(delta) <= 1 else float(k)


def analyse_frame(frame, fs=SAMPLE_RATE, a4=440.0, rms_threshold=RMS_THRESHOLD):
    """Analyse un bloc audio et retourne les informations sur la note détectée."""
    frame = np.asarray(frame, dtype=float)
    # On enlève la moyenne pour centrer le signal sur 0
    frame = frame - frame.mean()
    # Calcul de la valeur RMS pour mesurer le volume
    rms = float(np.sqrt(np.mean(frame ** 2)))
    freqs, mag = spectrum(frame, fs)
    result = {"freqs": freqs, "mag": mag, "rms": rms, "voiced": False,
              "f0": None, "note": None, "octave": None, "cents": None, "target": None}
    # Si le son est trop faible, on s'arrête là (silence)
    if rms < rms_threshold:
        return result
    # Recherche du pic principal
    k = hps_peak(mag, freqs)
    # Affinage du pic
    k_exact = parabolic_interp(mag, k)
    # Conversion de l'indice en fréquence (Hz)
    f0 = k_exact * fs / len(frame)
    # Détermination de la note correspondante
    name, octave, cents, target = freq_to_note(f0, a4)
    result.update(voiced=True, f0=f0, note=name, octave=octave, cents=cents, target=target)
    return result
