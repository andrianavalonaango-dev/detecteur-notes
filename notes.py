"""Conversion d'une fréquence en note de musique."""
import math

# Noms des 12 notes de la gamme chromatique
NOTES_FR = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


def freq_to_note(freq, a4=440.0):
    """Renvoie le nom de la note, l'octave, l'écart en cents et la fréquence idéale."""
    # Conversion de la fréquence en numéro de note MIDI
    midi = 69 + 12 * math.log2(freq / a4)
    # Note la plus proche (valeur entière)
    n = round(midi)
    # Calcul du décalage par rapport à la note juste (1 demi-ton = 100 cents)
    cents = 100 * (midi - n)
    # Nom de la note (Do, Ré, Mi...) et calcul de l'octave
    name = NOTES_FR[n % 12]
    octave = n // 12 - 1
    # Fréquence exacte que la note devrait avoir
    target = a4 * 2 ** ((n - 69) / 12)
    return name, octave, cents, target
