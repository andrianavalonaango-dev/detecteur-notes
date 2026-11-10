"""Tests unitaires pour valider les fonctions de calcul et de robustesse."""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dft import dft_naive, fft_radix2, rfft_radix2
from main import read_wav, synth
from notes import freq_to_note
from pitch import (
    FRAME_SIZE,
    SAMPLE_RATE,
    PitchTracker,
    analyse_frame,
)

# --- Tests existants originaux conservés ---


def test_exemple_du_cours():
    """Test avec un petit vecteur pour vérifier la formule de la DFT."""
    x = [1, 6, 8, -4, 9]
    assert np.allclose(dft_naive(x), np.fft.fft(x))
    # Le premier terme X[0] doit être égal à la somme des éléments de x
    assert np.isclose(dft_naive(x)[0], 20)


def test_fft_equals_dft():
    """Vérifie que la FFT rapide donne le même résultat que la DFT naïve."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(64)
    assert np.allclose(fft_radix2(x), dft_naive(x))
    assert np.allclose(fft_radix2(x), np.fft.fft(x))


def test_notes():
    """Vérifie que les fréquences correspondent aux bonnes notes."""
    assert freq_to_note(440)[:2] == ("La", 4)
    assert freq_to_note(261.63)[:2] == ("Do", 4)
    assert abs(freq_to_note(440 * 2 ** (10 / 1200))[2] - 10) < 1e-6


def test_detection_notes_instrument():
    """Test de détection de note sur différentes fréquences harmoniques."""
    for f in [82.41, 110.0, 196.0, 261.63, 329.63, 440.0, 880.0]:
        r = analyse_frame(synth(f, dur=0.5)[:FRAME_SIZE], SAMPLE_RATE)
        assert abs(r["f0"] - f) < 0.5, (f, r["f0"])


def test_sinus_pur():
    """Test avec une onde sinusoïdale pure (sans harmoniques)."""
    r = analyse_frame(synth(440, harmonics=(1.0,))[:FRAME_SIZE], SAMPLE_RATE)
    assert abs(r["f0"] - 440) < 0.5


def test_silence():
    """Vérifie qu'aucun son n'est détecté si le signal est vide."""
    assert not analyse_frame(np.zeros(FRAME_SIZE))["voiced"]


# --- Nouveaux tests de qualité, précision et robustesse ---


def test_precision_notes_graves():
    """Vérifie la précision (< 1 cent) sur les notes graves (Mi2, Do2)."""
    notes_graves = [
        ("Do2", 65.41),
        ("Mi2", 82.41),
        ("La2", 110.0),
        ("Do3", 130.81),
    ]
    for _name, f_true in notes_graves:
        sig = synth(f_true, dur=0.5)[:FRAME_SIZE]
        r = analyse_frame(sig, SAMPLE_RATE)
        assert r["voiced"]
        # Erreur en cents = 1200 * log2(f_est / f_true)
        err_cents = abs(1200.0 * np.log2(r["f0"] / f_true))
        msg = f"Erreur trop élevée pour {f_true} Hz: {err_cents} cents"
        assert err_cents < 1.0, msg


def test_signal_bruite():
    """Vérifie la robustesse en présence de bruit blanc (erreur < 5 cents)."""
    rng = np.random.default_rng(123)
    for f_true in [82.41, 196.0, 440.0, 880.0]:
        sig = synth(f_true, dur=0.5)[:FRAME_SIZE]
        # Ajout de bruit gaussien (SNR ~ 20 dB)
        noise = 0.03 * rng.standard_normal(FRAME_SIZE)
        r = analyse_frame(sig + noise, SAMPLE_RATE)
        assert r["voiced"]
        err_cents = abs(1200.0 * np.log2(r["f0"] / f_true))
        msg = f"Erreur avec bruit pour {f_true} Hz: {err_cents} cents"
        assert err_cents < 5.0, msg


def test_fondamentale_affaiblie():
    """Vérifie que l'HPS retrouve f0 quand la fondamentale est faible."""
    t = np.arange(FRAME_SIZE, dtype=float) / SAMPLE_RATE
    f0 = 110.0  # La2
    # Fondamentale affaiblie (0.2), harmoniques 2 et 3 prédominantes
    sig = (
        0.2 * np.sin(2.0 * np.pi * f0 * t)
        + 1.0 * np.sin(2.0 * np.pi * 2 * f0 * t)
        + 0.7 * np.sin(2.0 * np.pi * 3 * f0 * t)
        + 0.4 * np.sin(2.0 * np.pi * 4 * f0 * t)
    )
    r = analyse_frame(sig, SAMPLE_RATE)
    assert r["voiced"]
    assert abs(r["f0"] - f0) < 0.5, f"Fondamentale manquée: {r['f0']} != {f0}"
    assert r["note"] == "La" and r["octave"] == 2


def test_plage_frequences_60_1500_hz():
    """Vérifie la détection sur toute la plage utile musicale (60-1500 Hz)."""
    frequences = [
        65.41,    # Do2
        130.81,   # Do3
        261.63,   # Do4
        523.25,   # Do5
        1046.50,  # Do6
        1479.98,  # Fa#6
    ]
    for f in frequences:
        sig = synth(f, dur=0.5)[:FRAME_SIZE]
        r = analyse_frame(sig, SAMPLE_RATE)
        assert r["voiced"]
        err_cents = abs(1200.0 * np.log2(r["f0"] / f))
        assert err_cents < 1.0, f"Fréquence {f} Hz err: {err_cents} cents"


def test_wav_stereo(tmp_path):
    """Vérifie la lecture et conversion automatique en mono d'un WAV stéréo."""
    wav_file = tmp_path / "test_stereo.wav"
    fs = 44100
    t = np.arange(fs // 2, dtype=float) / fs
    sig_l = (0.5 * np.sin(2.0 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    sig_r = (0.5 * np.sin(2.0 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    interleaved = np.empty((len(t) * 2,), dtype=np.int16)
    interleaved[0::2] = sig_l
    interleaved[1::2] = sig_r

    with wave.open(str(wav_file), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(interleaved.tobytes())

    data, read_fs = read_wav(str(wav_file))
    assert read_fs == fs
    assert data.ndim == 1
    assert len(data) == len(t)
    # Détection de la note sur le signal extrait
    r = analyse_frame(data[:FRAME_SIZE], read_fs)
    assert r["voiced"]
    assert abs(r["f0"] - 440.0) < 0.2


def test_pitch_tracker_lissage_et_hysteresis():
    """Vérifie le lissage médian et l'hystérésis de PitchTracker."""
    tracker = PitchTracker(a4=440.0, hysteresis_cents=6.0)

    # 1) Lissage médian sur 5 trames avec une valeur aberrante isolée
    f_sequence = [440.0, 440.2, 880.0, 440.1, 439.9]
    last_res = None
    for f in f_sequence:
        dummy_result = {
            "freqs": np.array([440.0]),
            "mag": np.array([1.0]),
            "rms": 0.05,
            "voiced": True,
            "f0": f,
            "note": "La",
            "octave": 4,
            "cents": 0.0,
            "target": 440.0,
        }
        last_res = tracker.process(dummy_result, rms=0.05)

    # La médiane élimine le pic aberrant à 880 Hz
    assert abs(last_res["f0"] - 440.0) < 0.5

    # 2) Hystérésis : note limite près de 50 cents (Do4 / Do#4 à ~269.3 Hz)
    tracker.reset()
    res1 = tracker.process(
        {
            "freqs": np.array([269.2]),
            "mag": np.array([1.0]),
            "rms": 0.05,
            "voiced": True,
            "f0": 269.2,
            "note": "Do",
            "octave": 4,
            "cents": 49.5,
            "target": 261.63,
        },
        rms=0.05,
    )
    assert res1["note"] == "Do"

    # Légère variation franchissant 50 cents (+50.5 cents)
    res2 = tracker.process(
        {
            "freqs": np.array([269.4]),
            "mag": np.array([1.0]),
            "rms": 0.05,
            "voiced": True,
            "f0": 269.4,
            "note": "Do#",
            "octave": 4,
            "cents": -49.5,
            "target": 277.18,
        },
        rms=0.05,
    )
    # Grâce à l'hystérésis (marge de 6 cents), la note reste stable sur Do4
    assert res2["note"] == "Do"


def test_fft_radix2_signaux_complexes():
    """Vérifie la FFT radix-2 sur signaux complexes de différentes tailles."""
    rng = np.random.default_rng(99)
    for n in [8, 16, 64, 256, 1024]:
        xc = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        assert np.allclose(fft_radix2(xc), np.fft.fft(xc))
        xr = rng.standard_normal(n)
        assert np.allclose(rfft_radix2(xr), np.fft.rfft(xr))
