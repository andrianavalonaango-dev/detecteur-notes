"""Programme principal pour l'accordeur et le détecteur de notes en temps réel.

Prend en charge :
- L'accordeur graphique en direct (--live) avec affichage fluide (blit)
- L'accordeur console interactif (--cli)
- L'analyse d'un fichier audio WAV 16 bits (--wav)
- La génération et détection d'un son synthétique pur ou harmonique (--demo)
- La sélection du périphérique audio (--device) et du diapason La4 (--a4)
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import wave

import numpy as np

from pitch import FRAME_SIZE, SAMPLE_RATE, PitchTracker, analyse_frame

# --- Fonctions utilitaires ---


def synth(
    freq: float,
    fs: int = SAMPLE_RATE,
    dur: float = 1.0,
    harmonics: tuple[float, ...] = (1.0, 0.5, 0.3, 0.15),
) -> np.ndarray:
    """Génère un signal de test harmonique normalisé."""
    t = np.arange(int(fs * dur), dtype=float) / fs
    sig = sum(
        a * np.sin(2.0 * np.pi * freq * (i + 1) * t)
        for i, a in enumerate(harmonics)
    )
    max_amp = max(harmonics) if harmonics else 1.0
    return 0.3 * (sig / max_amp)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Lit un fichier audio WAV (PCM 16 bits) mono ou stéréo sans traceback."""
    try:
        with wave.open(path, "rb") as w:
            fs = w.getframerate()
            ch = w.getnchannels()
            width = w.getsampwidth()
            n_frames = w.getnframes()
            if width != 2:
                print(
                    f"Erreur : le fichier '{path}' utilise un codage "
                    f"sur {width * 8} bits.\n"
                    "Seuls les fichiers WAV PCM 16 bits sont supportés.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if n_frames == 0:
                print(
                    f"Erreur : le fichier '{path}' est vide.",
                    file=sys.stderr,
                )
                sys.exit(1)
            raw = w.readframes(n_frames)
    except FileNotFoundError:
        print(
            f"Erreur : le fichier '{path}' est introuvable.",
            file=sys.stderr,
        )
        sys.exit(1)
    except wave.Error as e:
        print(
            f"Erreur lors de la lecture du fichier WAV '{path}' : {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except OSError as e:
        print(
            f"Erreur d'accès au fichier '{path}' : {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    data = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768.0
    # Conversion stéréo (ou multicanal) en mono par moyennage
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, fs


def format_result(r: dict) -> str:
    """Formate le résultat d'analyse pour l'affichage console."""
    if not r.get("voiced") or r.get("f0") is None:
        return "  --  (silence)"
    sign = "+" if r["cents"] >= 0 else ""
    return (
        f"{r['note']:<4}{r['octave']}  {r['f0']:8.2f} Hz  "
        f"({sign}{r['cents']:.1f} cents)"
    )


def pad_frame(frame: np.ndarray) -> np.ndarray:
    """Complète le bloc par des zéros si sa taille est inférieure."""
    if len(frame) < FRAME_SIZE:
        return np.pad(frame, (0, FRAME_SIZE - len(frame)))
    return frame


# --- Analyse d'un flux audio ou d'un fichier ---


def run_signal(
    sig: np.ndarray,
    fs: int,
    a4: float = 440.0,
    hop: int = FRAME_SIZE // 2,
) -> None:
    """Analyse un signal audio par blocs successifs avec lissage temporel."""
    tracker = PitchTracker(a4=a4)
    for start in range(0, max(1, len(sig) - FRAME_SIZE + 1), hop):
        block = pad_frame(sig[start:start + FRAME_SIZE])
        r = analyse_frame(block, fs, a4, tracker=tracker)
        print(f"t={start / fs:6.2f}s  {format_result(r)}")


# --- Gestion du microphone et tampon circulaire sans np.roll ---


class MicBuffer:
    """Tampon circulaire pour capturer le flux micro en continu.

    Remplace l'ancien np.roll (qui déplaçait 8192 floats à chaque callback)
    par une écriture directe à pointeur tournant.
    """

    def __init__(self, fs: int, device: str | int | None = None) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            print(
                "Erreur : sounddevice est requis pour la capture microphone.",
                file=sys.stderr,
            )
            sys.exit(1)

        self._sd = sd
        self.size = FRAME_SIZE
        self.buf_data = np.zeros(self.size, dtype=float)
        self.write_pos = 0

        # Résolution du périphérique si fourni sous forme de chaîne numérique
        dev_id: int | str | None = None
        if device is not None:
            dev_id = int(device) if str(device).isdigit() else device

        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=fs,
                blocksize=1024,
                device=dev_id,
                callback=self._callback,
            )
        except (sd.PortAudioError, OSError, ValueError, TypeError) as err:
            print(
                f"Erreur d'accès au microphone : {err}\n"
                "Vérifiez qu'un micro est branché ou indiquez un "
                "périphérique avec l'option --device.",
                file=sys.stderr,
            )
            sys.exit(1)

    def _callback(self, indata, frames, time_info, status):
        """Callback audio : insertion directe sans décalage np.roll."""
        if status:
            pass
        chunk = indata[:, 0]
        n = len(chunk)
        if n >= self.size:
            self.buf_data[:] = chunk[-self.size:]
            self.write_pos = 0
            return

        end = self.write_pos + n
        if end <= self.size:
            self.buf_data[self.write_pos:end] = chunk
        else:
            first = self.size - self.write_pos
            self.buf_data[self.write_pos:] = chunk[:first]
            self.buf_data[:end - self.size] = chunk[first:]
        self.write_pos = end % self.size

    def get_frame(self) -> np.ndarray:
        """Retourne les FRAME_SIZE derniers échantillons chronologiques."""
        if self.write_pos == 0:
            return self.buf_data.copy()
        return np.concatenate(
            (self.buf_data[self.write_pos:], self.buf_data[:self.write_pos])
        )

    @property
    def buf(self) -> np.ndarray:
        """Propriété de rétrocompatibilité pour accéder au signal ordonné."""
        return self.get_frame()

    def __enter__(self):
        try:
            self.stream.start()
        except (self._sd.PortAudioError, OSError) as err:
            print(
                f"Erreur lors du démarrage du flux audio : {err}",
                file=sys.stderr,
            )
            sys.exit(1)
        return self

    def __exit__(self, *args):
        with contextlib.suppress(self._sd.PortAudioError, OSError):
            self.stream.stop()
            self.stream.close()


def run_cli(
    fs: int, a4: float = 440.0, device: str | int | None = None
) -> None:
    """Mode console : écoute le micro et affiche la note en direct."""
    tracker = PitchTracker(a4=a4)
    with MicBuffer(fs, device=device) as mic:
        print("Écoute... (Ctrl+C pour quitter)")
        try:
            while True:
                r = analyse_frame(mic.get_frame(), fs, a4, tracker=tracker)
                line = "\r" + format_result(r).ljust(50)
                print(line, end="", flush=True)
                time.sleep(0.08)
        except KeyboardInterrupt:
            print()


def run_gui(
    fs: int, a4: float = 440.0, device: str | int | None = None
) -> None:
    """Mode graphique : affiche le spectre et l'aiguille avec blit accéléré."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax_spec, ax_tune) = plt.subplots(
        2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1.2]}
    )
    fig.suptitle("Accordeur chromatique - DFT temps réel")
    (line,) = ax_spec.plot([], [], lw=1, color="#1f77b4")
    peak = ax_spec.axvline(0, color="red", ls="--", lw=1.2, visible=False)
    ax_spec.set(
        xlim=(0, 2000),
        ylim=(0, 0.25),
        xlabel="Fréquence (Hz)",
        ylabel="Amplitude |X[k]|",
    )
    ax_spec.grid(alpha=0.3)
    title = ax_spec.text(
        0.98,
        0.9,
        "",
        transform=ax_spec.transAxes,
        ha="right",
        fontsize=22,
        weight="bold",
    )
    ax_tune.set(
        xlim=(-50, 50),
        ylim=(0, 1),
        yticks=[],
        xlabel="Écart (cents)  <- trop bas | trop haut ->",
    )
    ax_tune.axvline(0, color="green", lw=2)
    (needle,) = ax_tune.plot([0, 0], [0, 1], lw=6, color="gray")

    tracker = PitchTracker(a4=a4)

    def init():
        line.set_data([], [])
        peak.set_visible(False)
        title.set_text("")
        needle.set_data([0, 0], [0, 1])
        needle.set_color("gray")
        return line, peak, title, needle

    def update(_):
        r = analyse_frame(mic.get_frame(), fs, a4, tracker=tracker)
        line.set_data(r["freqs"], r["mag"])
        if r["voiced"] and r["f0"] is not None:
            f0 = r["f0"]
            peak.set_xdata([f0, f0])
            peak.set_visible(True)
            title.set_text(f"{r['note']}{r['octave']}  {f0:.1f} Hz")
            c = max(-50.0, min(50.0, r["cents"]))
            needle.set_data([c, c], [0, 1])
            col = (
                "green"
                if abs(c) < 5
                else "orange"
                if abs(c) < 20
                else "red"
            )
            needle.set_color(col)
        else:
            peak.set_visible(False)
            title.set_text("--")
            needle.set_data([0, 0], [0, 1])
            needle.set_color("gray")
        return line, peak, title, needle

    with MicBuffer(fs, device=device) as mic:
        fig._anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            interval=60,
            blit=True,
            cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show()


# --- Lancement du programme ---


def main() -> None:
    """Analyse des arguments de la ligne de commande et exécution."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--live", action="store_true", help="interface graphique temps réel"
    )
    g.add_argument(
        "--cli", action="store_true", help="affichage console temps réel"
    )
    g.add_argument(
        "--wav", metavar="FICHIER", help="analyser un fichier WAV 16 bits"
    )
    g.add_argument(
        "--demo",
        type=float,
        metavar="HZ",
        help="signal synthétique à cette fréquence",
    )
    p.add_argument(
        "--a4",
        type=float,
        default=440.0,
        help="fréquence de référence du La4 (défaut 440)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="nom ou index du périphérique audio (microphone)",
    )
    args = p.parse_args()

    if args.live:
        run_gui(SAMPLE_RATE, args.a4, device=args.device)
    elif args.cli:
        run_cli(SAMPLE_RATE, args.a4, device=args.device)
    elif args.wav:
        sig, fs = read_wav(args.wav)
        run_signal(sig, fs, args.a4)
    else:
        run_signal(synth(args.demo, dur=0.5), SAMPLE_RATE, args.a4)


if __name__ == "__main__":
    main()
