from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, Preformatted

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10.5, leading=15, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceBefore=12, textColor=colors.HexColor("#1F3A5F"))
cap = ParagraphStyle("c", parent=body, fontSize=9, textColor=colors.grey, alignment=1)
code = ParagraphStyle("code", parent=ss["Code"], fontSize=8.5, leading=11, backColor=colors.HexColor("#F2F2F2"), borderPadding=4)

S = []
S += [Paragraph("Détecteur de notes et accordeur chromatique en temps réel", ParagraphStyle("t", parent=ss["Title"], fontSize=20)),
      Paragraph("Traitement du signal - Application de la Transformée de Fourier Discrète (Python)", ParagraphStyle("s", parent=body, alignment=1)),
      Paragraph("Étudiant : [Votre nom] &nbsp;&nbsp;|&nbsp;&nbsp; Date : 10 novembre 2026", ParagraphStyle("s2", parent=body, alignment=1)),
      Spacer(1, 10)]

S += [Paragraph("1. Objectif", h1),
      Paragraph("Développer un système qui capte un son au microphone, en extrait la fréquence fondamentale grâce à la "
                "Transformée de Fourier Discrète (DFT), puis affiche la note de musique correspondante et l'écart d'accordage "
                "en cents, comme un accordeur chromatique. Dépôt GitHub : [lien du dépôt].", body)]

S += [Paragraph("2. Architecture du système", h1)]
t = Table([["Étape", "Module", "Rôle"],
           ["Acquisition", "main.py (sounddevice)", "Capture à 44 100 Hz, tampon circulaire de 8192 échantillons"],
           ["Prétraitement", "pitch.py", "Suppression de la composante continue, porte de bruit (RMS), fenêtre de Hann"],
           ["DFT", "dft.py", "DFT naïve O(N<super>2</super>) et FFT radix-2 (Cooley-Tukey) codées à la main"],
           ["Fondamentale", "pitch.py", "Harmonic Product Spectrum + interpolation parabolique"],
           ["Note", "notes.py", "Fréquence vers note, octave et cents (La4 = 440 Hz)"],
           ["Affichage", "main.py (matplotlib)", "Spectre en direct, note détectée, aiguille d'accordage"]],
          colWidths=[3*cm, 4.5*cm, 9*cm])
t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F3A5F")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                       ("GRID", (0,0), (-1,-1), .4, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 9),
                       ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
t2 = Table([[Paragraph(c, ParagraphStyle("tc", parent=body, fontSize=9, leading=11, spaceAfter=0,
            textColor=colors.white if r == 0 else colors.black)) for c in row] for r, row in enumerate(t._cellvalues)],
           colWidths=[3*cm, 4.5*cm, 9*cm]); t2.setStyle(t._argW and TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F3A5F")),
                       ("GRID", (0,0), (-1,-1), .4, colors.grey), ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
S += [t2]

S += [Paragraph("3. La Transformée de Fourier Discrète", h1),
      Paragraph("Pour un signal échantillonné x[n] de N points, la DFT donne N coefficients complexes :", body),
      Paragraph("X[k] = somme pour n = 0 à N-1 de x[n] . exp(-i . 2 . pi . k . n / N),  k = 0 ... N-1", ParagraphStyle("f", parent=body, alignment=1, fontName="Helvetica-Oblique")),
      Paragraph("Le coefficient X[k] correspond à la fréquence f<sub>k</sub> = k . f<sub>s</sub> / N. Son module |X[k]| mesure la quantité d'énergie "
                "du signal à cette fréquence. Pour un signal réel, le spectre est symétrique : on ne garde que k = 0 ... N/2 (fréquences de 0 à f<sub>s</sub>/2). "
                "Avec f<sub>s</sub> = 44 100 Hz et N = 8192, la résolution est f<sub>s</sub>/N = 5,38 Hz.", body),
      Paragraph("Vérification : l'exemple du cours x = (1, 6, 8, -4, 9) donne X[0] = 1+6+8-4+9 = 20, et notre implémentation coïncide avec numpy.fft.fft "
                "(test unitaire <i>test_exemple_du_cours</i>). La FFT radix-2 donne exactement le même résultat que la DFT naïve mais en O(N log N), "
                "ce qui est indispensable pour le temps réel (8192<super>2</super> = 67 millions d'opérations contre environ 100 000).", body)]

S += [Paragraph("4. Extraction de la fréquence fondamentale", h1),
      Paragraph("<b>a) Fenêtrage.</b> Une trame est un morceau de signal coupé brutalement, ce qui étale l'énergie du pic sur les bins voisins "
                "(fuites spectrales). On multiplie la trame par une fenêtre de Hann pour obtenir des pics nets.", body),
      Paragraph("<b>b) Spectre.</b> On calcule |X[k]| par la DFT. Un instrument produit une fréquence fondamentale f<sub>0</sub> et des harmoniques "
                "2f<sub>0</sub>, 3f<sub>0</sub>, 4f<sub>0</sub>... visibles sur le diagramme du spectre (figure 1). Le pic le plus haut n'est pas toujours la fondamentale.", body),
      Paragraph("<b>c) Harmonic Product Spectrum.</b> On multiplie le spectre par ses versions sous-échantillonnées (facteurs 2, 3, 4) : "
                "HPS[k] = |X[k]| . |X[2k]| . |X[3k]| . |X[4k]|. Seule la vraie fondamentale a de l'énergie à la fois en k, 2k, 3k et 4k, "
                "ce qui évite les erreurs d'octave. Un seuil écarte les candidats sans énergie propre.", body),
      Paragraph("<b>d) Interpolation parabolique.</b> Le pic réel se trouve rarement exactement sur un bin. On ajuste une parabole sur le logarithme "
                "du spectre autour du bin maximal k : k<sub>exact</sub> = k + 0,5 (a - c) / (a - 2b + c), avec a, b, c les valeurs en k-1, k, k+1. "
                "La précision passe de 5 Hz à moins de 0,5 Hz.", body),
      Paragraph("<b>e) Fréquence et note.</b> f<sub>0</sub> = k<sub>exact</sub> . f<sub>s</sub> / N. Le numéro MIDI vaut m = 69 + 12 . log<sub>2</sub>(f<sub>0</sub>/440) ; "
                "la note est l'entier le plus proche, et l'écart en cents vaut 100 . (m - round(m)). L'accordeur est juste si |cents| &lt; 5.", body)]

S += [PageBreak(), Paragraph("5. Résultats", h1),
      Image("figures/spectre.png", width=15.5*cm, height=15.5*cm*6/9),
      Paragraph("Figure 1 - Signal temporel (haut) et spectre fréquentiel |X[k]| (bas). Fondamentale détectée : 442,05 Hz (La4, +8 cents). "
                "Les pics suivants sont les harmoniques 2f<sub>0</sub>, 3f<sub>0</sub>, 4f<sub>0</sub>.", cap),
      Spacer(1, 8),
      Image("figures/accordeur.png", width=15.5*cm, height=15.5*cm*6/9),
      Paragraph("Figure 2 - Capture de l'accordeur (mode --live) : spectre, note détectée et aiguille d'écart en cents.", cap)]

S += [Paragraph("6. Tests et validation", h1)]
rows = [["Fréquence jouée (Hz)", "Fréquence détectée (Hz)", "Note"],
        ["82,41", "82,49", "Mi2"], ["196,00", "196,04", "Sol3"], ["440,00", "439,91", "La4"], ["659,25", "659,29", "Mi5"]]
tt = Table(rows, colWidths=[5*cm, 5*cm, 3*cm])
tt.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F3A5F")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                        ("GRID", (0,0), (-1,-1), .4, colors.grey), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTSIZE", (0,0), (-1,-1), 9)]))
S += [tt, Spacer(1, 6),
      Paragraph("Signaux synthétiques riches en harmoniques avec bruit et fondamentale affaiblie : l'erreur reste inférieure à 0,1 Hz (2 cents). "
                "Six tests unitaires (pytest) valident la DFT, la FFT, la conversion en notes, la détection et le silence.", body)]

S += [Paragraph("7. Limites et améliorations", h1),
      Paragraph("La résolution (5,4 Hz) rend les basses fréquences moins précises avant interpolation ; on pourrait augmenter N ou ajouter du zero-padding. "
                "Les sons très bruités ou polyphoniques (accords) ne sont pas gérés. Une amélioration possible est un lissage temporel (médiane des dernières estimations).", body),
      Paragraph("8. Conclusion", h1),
      Paragraph("Le système détecte la note jouée en temps réel avec une précision de quelques cents en s'appuyant sur une DFT implémentée "
                "à la main, fenêtrée, puis affinée par HPS et interpolation parabolique.", body)]

SimpleDocTemplate("rapport_accordeur_DFT.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
                  title="Accordeur chromatique DFT").build(S)
