#!/usr/bin/env python3
"""
Combined baseline PSD figure across both baseline tests:
  - standalone device baseline (standalone_baseline_test)
  - breakout board baseline (crosstalk_test)

All 16 raw channel PSDs of each test are overlaid: standalone in blue,
breakout in red. On top of each set, the across-channel median per
frequency bin is plotted, smoothed with a moving-average filter (same
approach as each fig8).

Both tests share the same frequency grid (Welch, nperseg=2048, fs=1000),
so the traces overlay directly. PSD is in uV^2/Hz.

Usage:
    python3 plot_combined_baseline.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
BENCHTOP_ROOT = os.path.dirname(ROOT)
STANDALONE_INTERIM = os.path.join(BENCHTOP_ROOT, "standalone_baseline_test",
                                  "outputs", "interim_data.npz")
BREAKOUT_INTERIM = os.path.join(BENCHTOP_ROOT, "crosstalk_test",
                                "outputs", "interim_data.npz")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# Shared with the per-test plot scripts (PSD_YLIM, PSD_SMOOTH_BINS)
PSD_YLIM = (0.0, 2.5e-2)
PSD_SMOOTH_BINS = 21  # moving-average window (~10 Hz at 0.49 Hz resolution)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
})

os.makedirs(FIGURE_DIR, exist_ok=True)


def load_psd(interim_path):
    """Return (freqs, psd_matrix, median_psd_rms_uV) with one row per channel,
    PSD in uV^2/Hz. The RMS is integrated over the full PSD band (Parseval),
    median across channels."""
    d = np.load(interim_path, allow_pickle=True)
    freqs = np.asarray(d["noise_welch_freqs"][0], dtype=float)
    psd = np.vstack([np.asarray(x, dtype=float) for x in d["noise_welch_psd_uv2"]])
    df = freqs[1] - freqs[0]
    rms_per_ch = np.sqrt(np.sum(psd * df, axis=1))
    return freqs, psd, float(np.median(rms_per_ch))


def plot_combined_baseline(output_path):
    f_sa, psd_sa, rms_sa = load_psd(STANDALONE_INTERIM)
    f_bo, psd_bo, rms_bo = load_psd(BREAKOUT_INTERIM)
    if not np.allclose(f_sa, f_bo):
        raise RuntimeError("Frequency grids differ between the two interim files")

    # Smoothed central line per test: median across channels, then moving average
    med_sa = uniform_filter1d(np.median(psd_sa, axis=0), size=PSD_SMOOTH_BINS,
                              mode="nearest")
    med_bo = uniform_filter1d(np.median(psd_bo, axis=0), size=PSD_SMOOTH_BINS,
                              mode="nearest")

    fig, ax = plt.subplots(figsize=(7.0, 7.0 * 0.618))
    for row in psd_sa:
        ax.plot(f_sa, row, color="#2166AC", linewidth=0.5, alpha=0.45)
    for row in psd_bo:
        ax.plot(f_bo, row, color="#D62728", linewidth=0.5, alpha=0.45)
    ax.plot([], [], color="#2166AC", linewidth=1.5,
            label=f"Standalone (16 ch), RMS = {rms_sa:.2f} uV")
    ax.plot([], [], color="#D62728", linewidth=1.5,
            label=f"Breakout (16 ch), RMS = {rms_bo:.2f} uV")
    ax.plot(f_sa, med_sa, color="#08519C", linewidth=1.8, zorder=5,
            label="Standalone – median (smoothed)")
    ax.plot(f_bo, med_bo, color="#A50F15", linewidth=1.8, zorder=5,
            label="Breakout – median (smoothed)")
    for h in range(1, 11):
        ax.axvline(50 * h, color="red", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (uV$^2$/Hz)")
    ax.set_title("Combined Baseline PSD – Standalone vs Breakout", fontsize=10,
                 fontweight="bold")
    ax.set_xlim([0, 500])
    ax.set_ylim(PSD_YLIM)
    ax.legend(fontsize=7.5, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("Combined Baseline PSD Plot")
    print("=" * 70)
    plot_combined_baseline(os.path.join(FIGURE_DIR, "fig_combined_baseline_psd.png"))
    print("\n  Done.")


if __name__ == "__main__":
    main()