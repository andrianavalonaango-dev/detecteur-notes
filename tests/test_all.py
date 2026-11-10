"""Tests unitaires pour valider les fonctions de calcul."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dft import dft_naive, fft_radix2
from notes import freq_to_note
from pitch import analyse_frame, SAMPLE_RATE, FRAME_SIZE
from main import synth


def test_exemple_du_cours():
    # Test avec un petit vecteur pour vérifier la formule de la DFT
    x = [1, 6, 8, -4, 9]
    assert np.allclose(dft_naive(x), np.fft.fft(x))
    # Le premier terme X[0] doit être égal à la somme des éléments de x
    assert np.isclose(dft_naive(x)[0], 20)


def test_fft_equals_dft():
    # Vérifie que la FFT rapide donne le même résultat que la DFT avec boucles
    x = np.random.randn(64)
    assert np.allclose(fft_radix2(x), dft_naive(x))
    assert np.allclose(fft_radix2(x), np.fft.fft(x))


def test_notes():
    # Vérifie que les fréquences correspondent aux bonnes notes de musique
    assert freq_to_note(440)[:2] == ("La", 4)
    assert freq_to_note(261.63)[:2] == ("Do", 4)
    assert abs(freq_to_note(440 * 2 ** (10 / 1200))[2] - 10) < 1e-6


def test_detection_notes_instrument():
    # Test de la détection de note sur différentes fréquences avec harmoniques
    for f in [82.41, 110.0, 196.0, 261.63, 329.63, 440.0, 880.0]:
        r = analyse_frame(synth(f, dur=0.5)[:FRAME_SIZE], SAMPLE_RATE)
        assert abs(r["f0"] - f) < 0.5, (f, r["f0"])


def test_sinus_pur():
    # Test avec une onde sinusoïdale pure (sans harmoniques)
    r = analyse_frame(synth(440, harmonics=(1.0,))[:FRAME_SIZE], SAMPLE_RATE)
    assert abs(r["f0"] - 440) < 0.5


def test_silence():
    # Vérifie qu'aucun son n'est détecté si le signal est vide (silence)
    assert not analyse_frame(np.zeros(FRAME_SIZE))["voiced"]
