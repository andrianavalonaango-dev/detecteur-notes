# Détecteur de notes et accordeur chromatique

Ce projet détecte la fréquence fondamentale d'un son et la convertit en note de musique. Il propose un accordeur graphique en temps réel, un affichage console, l'analyse de fichiers WAV et un mode de démonstration sans micro.

L'analyse utilise une fenêtre de Hann, une FFT radix-2, la méthode HPS pour repérer la fondamentale et une interpolation parabolique pour affiner sa fréquence. Les résultats incluent la note, l'octave et l'écart en cents par rapport à la note juste.

## Prérequis et installation

Python 3 et un microphone sont nécessaires pour les modes en direct. L'analyse de fichiers et le mode démo ne nécessitent pas de microphone.

```bash
python -m venv .venv
```

Activez ensuite l'environnement virtuel :

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Installez les dépendances :

```bash
python -m pip install -r requirements.txt
```

Pour générer le rapport PDF, installez aussi ReportLab :

```bash
python -m pip install reportlab
```

## Utilisation

À exécuter depuis le répertoire du projet.

```bash
# Accordeur graphique en temps réel (microphone)
python main.py --live

# Accordeur en console
python main.py --cli

# Analyse d'un fichier WAV PCM 16 bits
python main.py --wav note.wav

# Démonstration avec un signal synthétique à 440 Hz, sans microphone
python main.py --demo 440

# Changer la fréquence de référence du La4 (440 Hz par défaut)
python main.py --demo 442 --a4 442
```

## Figures et rapport

Les figures sont facultatives pour utiliser l'accordeur. Pour les (re)générer, le dossier `figures/` doit exister :

```bash
python make_figures.py
```

Cette commande crée `figures/spectre.png` et `figures/accordeur.png`. On peut également lui fournir un fichier WAV PCM 16 bits :

```bash
python make_figures.py note.wav
```

Après avoir généré les figures, créez le rapport PDF avec :

```bash
python make_report.py
```

Le rapport est enregistré dans `rapport_accordeur_DFT.pdf`. Le script de rapport utilise les deux images générées dans `figures/`.

## Tests

Lancez les tests unitaires avec :

```bash
python -m pytest
```

Ils vérifient notamment la DFT et la FFT, la conversion fréquence-note, la détection de notes synthétiques et le traitement du silence.

## Fichiers du projet

| Fichier ou dossier | Rôle |
|---|---|
| `main.py` | Point d'entrée et modes live, console, WAV et démo |
| `dft.py` | DFT directe et FFT radix-2 |
| `pitch.py` | Analyse du spectre et détection de la fréquence fondamentale |
| `notes.py` | Conversion d'une fréquence en note, octave et cents |
| `make_figures.py` | Génération des figures du rapport |
| `make_report.py` | Génération du rapport PDF à partir des figures |
| `figures/` | Images utilisées par le rapport |
| `tests/` | Tests unitaires |
| `requirements.txt` | Dépendances Python de l'application et des tests |
