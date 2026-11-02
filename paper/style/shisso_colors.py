"""Shisso figure palette (WS-132). Import in every paper figure script: `import shisso_colors as sc; sc.use()`.
ACCENT is the site accent; CATEGORICAL is the accent plus four Okabe-Ito hues; `python3 shisso_colors.py` checks the palette
under a deuteranopia simulation (Machado 2009, severity 1.0) and prints the minimum pairwise CIE76 delta-E."""
from itertools import combinations
from pathlib import Path

ACCENT = "#1F5FBF"
CATEGORICAL = [ACCENT, "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
GREY = "#8A8F99"      # baselines, bars
DARKGREY = "#555B66"  # the "bar" line (always dashed)
BAND_ALPHA = 0.20     # CI bands

def use():
    import matplotlib.pyplot as plt
    plt.style.use(str(Path(__file__).with_name("shisso.mplstyle")))

def bar_line(ax, y, label=None):
    return ax.axhline(y, color=DARKGREY, ls="--", lw=1.0, label=label)

def band(ax, x, lo, hi, color=ACCENT):
    return ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0)

_DEUT = [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]

def _lab(hexc):
    rgb = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    sim = [max(0.0, min(1.0, sum(m * c for m, c in zip(row, lin)))) for row in _DEUT]
    x = 0.4124 * sim[0] + 0.3576 * sim[1] + 0.1805 * sim[2]
    y = 0.2126 * sim[0] + 0.7152 * sim[1] + 0.0722 * sim[2]
    z = 0.0193 * sim[0] + 0.1192 * sim[1] + 0.9505 * sim[2]
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(x / 0.9505), f(y), f(z / 1.089)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)

def min_deut_delta_e(cols=CATEGORICAL):
    return min(sum((a - b) ** 2 for a, b in zip(_lab(p), _lab(q))) ** 0.5 for p, q in combinations(cols, 2))

if __name__ == "__main__":
    print(f"min pairwise deltaE76 under deuteranopia, {len(CATEGORICAL)} colours: {min_deut_delta_e():.1f}")
