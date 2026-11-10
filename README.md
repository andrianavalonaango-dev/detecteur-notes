# Détecteur de notes & accordeur chromatique temps réel (DFT)

Projet de Traitement du Signal : détection de la note jouée (fréquence fondamentale) à partir du micro,
avec une Transformée de Fourier Discrète (DFT) implémentée en Python.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation
```bash
python main.py --live          # accordeur graphique temps réel (micro)
python main.py --cli           # version console
python main.py --wav note.wav  # analyse d'un fichier WAV 16 bits
python main.py --demo 440      # test sans micro (signal synthétique)
python make_figures.py         # figures du rapport
python -m pytest               # tests unitaires
```

## Structure
| Fichier | Rôle |
|---|---|
| `dft.py` | DFT naïve O(N²) (définition du cours) + FFT radix-2 (Cooley-Tukey) |
| `pitch.py` | fenêtre de Hann, spectre, HPS, interpolation parabolique |
| `notes.py` | fréquence -> note (La4 = 440 Hz) + écart en cents |
| `main.py` | capture audio (sounddevice), modes live / cli / wav / demo |
| `tests/` | validation de la DFT (exemple du cours) et de la détection |

## Principe
Trame de 8192 échantillons à 44 100 Hz -> fenêtre de Hann -> DFT -> |X[k]| -> Harmonic Product Spectrum
-> pic affiné par interpolation parabolique -> f0 = k·fs/N -> note = 69 + 12·log2(f0/440), écart en cents.
