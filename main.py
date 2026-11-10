"""Programme principal pour l'accordeur et le détecteur de notes."""
import argparse
import sys
import time
import wave

import numpy as np

from pitch import analyse_frame, SAMPLE_RATE, FRAME_SIZE


# --- Fonctions utilitaires ---

def synth(freq, fs=SAMPLE_RATE, dur=1.0, harmonics=(1.0, 0.5, 0.3, 0.15)):
    """Génère un son de test avec plusieurs harmoniques."""
    t = np.arange(int(fs * dur)) / fs
    sig = sum(a * np.sin(2 * np.pi * freq * (i + 1) * t) for i, a in enumerate(harmonics))
    return 0.3 * sig / max(harmonics)


def read_wav(path):
    """Lit un fichier audio au format WAV (16 bits)."""
    with wave.open(path, "rb") as w:
        fs, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        sys.exit("Seuls les WAV PCM 16 bits sont supportés.")
    data = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768
    # Si le son est stéréo, on le convertit en mono en faisant la moyenne des canaux
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, fs


def format_result(r):
    """Formate le résultat pour l'affichage textuel."""
    if not r["voiced"]:
        return "  --  (silence)"
    sign = "+" if r["cents"] >= 0 else ""
    return f"{r['note']:<4}{r['octave']}  {r['f0']:8.2f} Hz  ({sign}{r['cents']:.1f} cents)"


def pad_frame(frame):
    """Complète le bloc avec des zéros si sa taille est insuffisante."""
    return np.pad(frame, (0, FRAME_SIZE - len(frame))) if len(frame) < FRAME_SIZE else frame


# --- Analyse d'un signal audio ou d'un fichier ---

def run_signal(sig, fs, hop=FRAME_SIZE // 2):
    """Analyse un signal audio en avançant par blocs successifs."""
    for start in range(0, max(1, len(sig) - FRAME_SIZE + 1), hop):
        r = analyse_frame(pad_frame(sig[start:start + FRAME_SIZE]), fs)
        print(f"t={start / fs:6.2f}s  {format_result(r)}")


# --- Gestion du microphone ---

class MicBuffer:
    """Tampon pour stocker les échantillons audio reçus du micro en continu."""

    def __init__(self, fs):
        import sounddevice as sd
        self.buf = np.zeros(FRAME_SIZE)
        self.stream = sd.InputStream(channels=1, samplerate=fs, blocksize=1024,
                                     callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        # On décale les anciens échantillons pour ajouter les nouveaux à la fin
        n = len(indata)
        self.buf = np.roll(self.buf, -n)
        self.buf[-n:] = indata[:, 0]

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *a):
        self.stream.stop()
        self.stream.close()


def run_cli(fs):
    """Mode console : écoute le micro et affiche la note en direct."""
    with MicBuffer(fs) as mic:
        print("Écoute... (Ctrl+C pour quitter)")
        try:
            while True:
                print("\r" + format_result(analyse_frame(mic.buf.copy(), fs)).ljust(50), end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print()


def run_gui(fs, a4):
    """Mode graphique : affiche le spectre et l'aiguille de l'accordeur."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax_spec, ax_tune) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1.2]})
    fig.suptitle("Accordeur chromatique - DFT temps réel")
    line, = ax_spec.plot([], [], lw=1)
    peak = ax_spec.axvline(0, color="red", ls="--", lw=1, visible=False)
    ax_spec.set(xlim=(0, 2000), ylim=(0, 0.2), xlabel="Fréquence (Hz)", ylabel="Amplitude |X[k]|")
    ax_spec.grid(alpha=.3)
    title = ax_spec.text(0.98, 0.9, "", transform=ax_spec.transAxes, ha="right", fontsize=22, weight="bold")
    ax_tune.set(xlim=(-50, 50), ylim=(0, 1), yticks=[], xlabel="Écart (cents)  <- trop bas | trop haut ->")
    ax_tune.axvline(0, color="green", lw=2)
    needle, = ax_tune.plot([0, 0], [0, 1], lw=6, color="gray")
    history = []

    def update(_):
        # Analyse du bloc sonore actuel
        r = analyse_frame(mic.buf.copy(), fs, a4)
        line.set_data(r["freqs"], r["mag"])
        ax_spec.set_ylim(0, max(0.05, r["mag"].max() * 1.2))
        if r["voiced"]:
            history.append(r["f0"])
            del history[:-5]
            peak.set_xdata([r["f0"]] * 2)
            peak.set_visible(True)
            title.set_text(f"{r['note']}{r['octave']}  {r['f0']:.1f} Hz")
            c = max(-50, min(50, r["cents"]))
            needle.set_data([c, c], [0, 1])
            # Couleur de l'aiguille selon la précision de l'accordage
            needle.set_color("green" if abs(c) < 5 else "orange" if abs(c) < 20 else "red")
        else:
            peak.set_visible(False)
            title.set_text("--")
            needle.set_color("gray")
        return line, peak, title, needle

    with MicBuffer(fs) as mic:
        _anim = FuncAnimation(fig, update, interval=80, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()


# --- Lancement du programme ---

def main():
    """Analyse des arguments de la ligne de commande et exécution."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", action="store_true", help="interface graphique temps réel")
    g.add_argument("--cli", action="store_true", help="affichage console temps réel")
    g.add_argument("--wav", metavar="FICHIER", help="analyser un fichier WAV 16 bits")
    g.add_argument("--demo", type=float, metavar="HZ", help="signal synthétique à cette fréquence")
    p.add_argument("--a4", type=float, default=440.0, help="fréquence de référence du La4 (défaut 440)")
    args = p.parse_args()

    if args.live:
        run_gui(SAMPLE_RATE, args.a4)
    elif args.cli:
        run_cli(SAMPLE_RATE)
    elif args.wav:
        sig, fs = read_wav(args.wav)
        run_signal(sig, fs)
    else:
        run_signal(synth(args.demo, dur=0.5), SAMPLE_RATE)


if __name__ == "__main__":
    main()
