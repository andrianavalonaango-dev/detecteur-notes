"""Génération des figures pour le rapport."""

import sys

import matplotlib.pyplot as plt
import numpy as np

from main import read_wav, synth
from pitch import FRAME_SIZE, SAMPLE_RATE, analyse_frame

plt.switch_backend("Agg")

# Chargement d'un fichier WAV s'il est fourni, sinon son synthétique
if len(sys.argv) > 1:
    sig, fs = read_wav(sys.argv[1])
    frame = sig[:FRAME_SIZE]
else:
    fs = SAMPLE_RATE
    rng = np.random.default_rng(1)
    base_sig = synth(
        442.0, dur=0.5, harmonics=(1.0, 0.6, 0.35, 0.2)
    )[:FRAME_SIZE]
    frame = base_sig + (0.01 * rng.standard_normal(FRAME_SIZE))

r = analyse_frame(frame, fs)

# 1) Tracé du signal audio dans le temps et de son spectre en fréquences
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
t = np.arange(FRAME_SIZE) / fs * 1000
ax[0].plot(t[:600], frame[:600], lw=1)
ax[0].set(
    title="Signal audio capté (domaine temporel)",
    xlabel="Temps (ms)",
    ylabel="Amplitude",
)
ax[1].plot(r["freqs"], r["mag"], lw=1)
label_f0 = (
    f"Fondamentale f0 = {r['f0']:.2f} Hz ({r['note']}{r['octave']})"
)
ax[1].axvline(r["f0"], color="red", ls="--", label=label_f0)
for h in (2, 3, 4):
    ax[1].axvline(h * r["f0"], color="gray", ls=":", lw=0.8)
ax[1].set(
    xlim=(0, 2000),
    title="Spectre fréquentiel |X[k]| (DFT)",
    xlabel="Fréquence (Hz)",
    ylabel="Amplitude",
)
ax[1].legend()
ax[1].grid(alpha=0.3)
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/spectre.png", dpi=150)
plt.close()

# 2) Capture de l'accordeur graphique
fig, (a1, a2) = plt.subplots(
    2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1.2]}
)
fig.suptitle("Accordeur chromatique - DFT temps réel")
a1.plot(r["freqs"], r["mag"], lw=1)
a1.axvline(r["f0"], color="red", ls="--", lw=1)
a1.set(xlim=(0, 2000), xlabel="Fréquence (Hz)", ylabel="Amplitude |X[k]|")
a1.grid(alpha=0.3)
txt = f"{r['note']}{r['octave']}  {r['f0']:.1f} Hz"
a1.text(
    0.98,
    0.9,
    txt,
    transform=a1.transAxes,
    ha="right",
    fontsize=22,
    weight="bold",
)
a2.set(
    xlim=(-50, 50),
    ylim=(0, 1),
    yticks=[],
    xlabel="Écart (cents)  <- trop bas | trop haut ->",
)
a2.axvline(0, color="green", lw=2)
c = max(-50, min(50, r["cents"]))
a2.plot([c, c], [0, 1], lw=6, color="green" if abs(c) < 5 else "orange")
plt.tight_layout()
plt.savefig("figures/accordeur.png", dpi=150)
plt.close()
print(r["f0"], r["note"], r["octave"], round(r["cents"], 1))
