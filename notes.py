"""Conversion d'une fréquence en note de musique de la gamme chromatique."""

from __future__ import annotations

import math

# Noms des 12 notes de la gamme chromatique tempérée
NOTES_FR: list[str] = [
    "Do", "Do#", "Ré", "Ré#", "Mi", "Fa",
    "Fa#", "Sol", "Sol#", "La", "La#", "Si",
]


def freq_to_note(
    freq: float, a4: float = 440.0
) -> tuple[str, int, float, float]:
    """Convertit une fréquence en note, octave, écart en cents et cible idéale.

    Args:
        freq: Fréquence mesurée en Hertz.
        a4: Fréquence de référence du diapason La4 (par défaut 440.0 Hz).

    Returns:
        tuple[str, int, float, float]: (nom, octave, écart en cents, f_cible).
    """
    if freq <= 0.0 or a4 <= 0.0:
        return ("--", 0, 0.0, 0.0)

    # Conversion de la fréquence en numéro de note MIDI continu
    midi = 69.0 + 12.0 * math.log2(freq / a4)
    # Numéro MIDI entier le plus proche
    n = round(midi)
    # Décalage en cents par rapport au tempérament (1 demi-ton = 100 cents)
    cents = 100.0 * (midi - n)
    # Nom de la note et octave scientifique correspondante
    name = NOTES_FR[n % 12]
    octave = (n // 12) - 1
    # Fréquence théorique exacte de la note cible
    target = a4 * (2.0 ** ((n - 69) / 12.0))
    return name, octave, cents, target
