# Détecteur de notes et accordeur chromatique

Ce projet analyse un son pour en détecter la fréquence fondamentale et afficher la note la plus proche, son octave et son écart en cents. Il propose un accordeur graphique en temps réel, un mode console, l'analyse de fichiers WAV et un mode de démonstration sans microphone.

L'analyse utilise une fenêtre de Hann, une FFT radix-2 (avec une variante optimisée pour les signaux réels), un zero-padding pour affiner la résolution fréquentielle, la méthode HPS pour repérer la fondamentale et une interpolation parabolique. En écoute continue, un suivi temporel lisse les estimations et limite les sauts de note.

## Prérequis

- Python 3.10 ou plus récent.
- Un microphone pour les modes en direct.
- Les fichiers WAV doivent être au format PCM 16 bits (mono ou stéréo).

## Installation

Depuis le répertoire du projet, créez un environnement virtuel :

```bash
python -m venv .venv
```

Activez-le :

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Installez les dépendances de l'application et des tests :

```bash
python -m pip install -r requirements.txt
```

Pour générer le rapport PDF, installez également ReportLab :

```bash
python -m pip install reportlab
```

## Utilisation

Les options de mode sont mutuellement exclusives :

```bash
# Accordeur graphique en temps réel
python main.py --live

# Accordeur en console
python main.py --cli

# Analyser un fichier WAV PCM 16 bits
python main.py --wav note.wav

# Démonstration avec un signal synthétique à 440 Hz, sans microphone
python main.py --demo 440
```

Le La4 est réglé à 440 Hz par défaut. Utilisez `--a4` pour choisir un autre diapason. Avec `--device`, vous pouvez préciser le nom ou l'index du microphone :

```bash
python main.py --live --a4 442 --device 1
python main.py --cli --device "Nom du microphone"
python main.py --demo 442 --a4 442
```

## Figures et rapport PDF

Le dossier `figures/` doit exister avant de générer les images. Depuis le répertoire du projet :

```bash
python make_figures.py
```

Cette commande produit `figures/spectre.png` et `figures/accordeur.png` à partir d'un signal synthétique. Pour générer les figures à partir d'un WAV :

```bash
python make_figures.py note.wav
```

Une fois les deux images générées, créez le rapport :

```bash
python make_report.py
```

Le rapport est enregistré dans `rapport_accordeur_DFT.pdf`. La génération du rapport dépend de ReportLab et des deux images dans `figures/`.

## Tests

Exécutez les tests unitaires avec :

```bash
python -m pytest
```

Les tests couvrent notamment la DFT/FFT pour signaux réels et complexes, la conversion fréquence-note, la détection sur des signaux synthétiques et bruités, le lissage temporel et la lecture WAV stéréo.

## Structure du projet

| Fichier ou dossier | Rôle |
|---|---|
| `main.py` | Point d'entrée : modes graphique, console, WAV et démo |
| `dft.py` | DFT directe, FFT radix-2 et FFT réelle radix-2 |
| `pitch.py` | Spectre, détection HPS et suivi temporel de la note |
| `notes.py` | Conversion d'une fréquence en note, octave, cents et fréquence cible |
| `make_figures.py` | Génération des images du rapport |
| `make_report.py` | Création du rapport PDF à partir des images |
| `figures/` | Images utilisées par le rapport |
| `tests/` | Tests unitaires |
| `requirements.txt` | Dépendances Python de l'application et des tests |
