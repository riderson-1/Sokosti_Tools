#!/usr/bin/env python3
"""
Plotting stage of the standalone-device baseline analysis.

Reads interim data from analyze.py and reproduces the reference baseline
figures/tables from the crosstalk pipeline:
  - fig7_baseline_noise: per-channel RMS noise bar chart (both bands)
  - fig8_baseline_psd:   PSD overlay of all 16 channels (nV/sqrt(Hz))
  - fig9_baseline_table: per-channel baseline table (CSV + LaTeX)

Usage:
    python3 plot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Paths
OUTPUT_DIR = "/home/karl/repos/Sokosti_tools/benchtop_testing_tools/standalone_baseline_test/outputs"
INTERIM_PATH = os.path.join(OUTPUT_DIR, "interim_data.npz")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

# Constants
N_CHANNELS = 16
# Shared with crosstalk_test/plot.py so both baseline PSD figures use the
# same y-axis (scale and limits) and are directly comparable.
PSD_YLIM = (0.0, 2.5e-2)
PSD_SMOOTH_BINS = 21  # moving-average window (~10 Hz at 0.49 Hz resolution)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
})

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def _fig_size(w, r=0.618):
    return (w, w * r)


def load_interim():
    """Load interim data from analyze.py."""
    d = np.load(INTERIM_PATH, allow_pickle=True)
    noise_ref = {}
    for i, ch in enumerate(d["noise_ch"]):
        ch = int(ch)
        noise_ref[ch] = {
            "rms_uV": float(d["noise_rms_uV"][i]),
            "welch_freqs_hz": d["noise_welch_freqs"][i],
            "welch_psd_uv2": d["noise_welch_psd_uv2"][i],
            "mean_uV": float(d["noise_mean_uV"][i]),
            "vpp_uV": float(d["noise_vpp_uV"][i]),
        }
    gap_info = d["gap_info"].item()
    if not isinstance(gap_info, dict):
        gap_info = dict(gap_info)
    return noise_ref, gap_info, float(d["duration_s"])


def plot_baseline_noise(noise_ref, output_path):
    """Per-channel whole-spectrum RMS noise bar chart (0 to Nyquist)."""
    channels = list(range(1, 17))
    rms = [noise_ref[ch]["rms_uV"] for ch in channels]
    fig, ax = plt.subplots(figsize=_fig_size(7.0, 0.5))
    x = np.arange(len(channels))
    ax.bar(x, rms, 0.6, color="#4472C4", label="RMS (0-500 Hz)")
    ax.axvline(7.5, color="black", linestyle="--", linewidth=1, label="ADS1299 boundary")
    ax.set_xlabel("Channel")
    ax.set_ylabel("RMS noise (uV)")
    ax.set_title("Per-Channel Noise Floor (Standalone Device Baseline)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in channels])
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_baseline_psd(noise_ref, output_path):
    """PSD overlay of all 16 channels plus a smoothed central-tendency line.

    The central line is the across-channel median per frequency bin (the
    middle of the distribution), smoothed with a moving-average filter
    along frequency. Y-scale and y-limits match the crosstalk tool's
    baseline PSD figure (PSD_YLIM)."""
    fig, ax = plt.subplots(figsize=_fig_size(7.0, 0.6))
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    freqs = None
    densities = []
    for ch in range(1, 17):
        nr = noise_ref[ch]
        freqs = nr["welch_freqs_hz"]
        densities.append(np.asarray(nr["welch_psd_uv2"], dtype=float))
        ax.plot(freqs, densities[-1], color=colors[ch - 1],
                label=f"CH{ch}", linewidth=0.7)
    # Middle of the across-channel distribution at each frequency,
    # smoothed with a moving-average filter.
    median_psd = np.median(np.vstack(densities), axis=0)
    median_smooth = uniform_filter1d(median_psd, size=PSD_SMOOTH_BINS,
                                     mode="nearest")
    ax.plot(freqs, median_smooth, color="black", linewidth=1.8, zorder=5,
            label="Median (smoothed)")
    for h in range(1, 11):
        ax.axvline(50 * h, color="red", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (uV$^2$/Hz)")
    ax.set_title("Standalone Device Baseline PSD – All 16 Channels", fontsize=10, fontweight="bold")
    ax.set_xlim([0, 500])
    ax.set_yscale("linear")
    ax.set_ylim(PSD_YLIM)
    ax.legend(fontsize=5.5, ncol=4, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def export_baseline_table(noise_ref, output_path):
    """Baseline table per channel + Median row (whole-spectrum RMS)."""
    channels = list(range(1, 17))
    rows = []
    for ch in channels:
        nr = noise_ref[ch]
        rows.append({
            "Channel": ch,
            "RMS (uV)": round(nr["rms_uV"], 2),
            "Vpp (uV)": round(nr["vpp_uV"], 1),
        })
    df = pd.DataFrame(rows)
    medians = {
        "Channel": "Median",
        "RMS (uV)": round(df["RMS (uV)"].median(), 2),
        "Vpp (uV)": round(df["Vpp (uV)"].median(), 1),
    }
    df = pd.concat([df, pd.DataFrame([medians])], ignore_index=True)
    fig, ax = plt.subplots(figsize=_fig_size(7.0, 0.5))
    ax.axis("off")

    numeric_cols = [1, 2]
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    for i in range(len(df)):
        for j in range(len(df.columns)):
            if i == 0:
                table[i, j].set_facecolor("#4472C4")
                table[i, j].set_text_props(color="white", fontweight="bold")
            elif j in numeric_cols and i < len(df) - 1:
                table[i, j].set_facecolor("white")
                table[i, j].set_text_props(color="black")
            elif i == len(df):
                table[i, j].set_facecolor("#D9E2F3")
                table[i, j].set_text_props(fontweight="bold", color="black")
    ax.set_title("Standalone Device Baseline Noise per Channel", fontsize=10, fontweight="bold", pad=50)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig9_baseline_table.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(os.path.join(FIGURE_DIR, "fig9_baseline_table.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: fig9_baseline_table.png/pdf")
    df.to_csv(os.path.join(TABLE_DIR, "baseline_channels.csv"), index=False)
    print(f"  Saved: {os.path.join(TABLE_DIR, 'baseline_channels.csv')}")
    latex = df.to_latex(index=False, escape=False, column_format="l" * len(df.columns))
    with open(os.path.join(TABLE_DIR, "baseline_channels.tex"), "w") as f:
        f.write(latex)
    print(f"  Saved: {os.path.join(TABLE_DIR, 'baseline_channels.tex')}")
    return df


def main():
    print("=" * 70)
    print("Plotting Stage: Standalone Device Baseline")
    print("=" * 70)

    noise_ref, gap_info, duration = load_interim()
    print(f"\n  Loaded interim data ({len(noise_ref)} channels, "
          f"gap-free duration {duration:.2f} s)")

    plot_baseline_noise(noise_ref, os.path.join(FIGURE_DIR, "fig7_baseline_noise.png"))
    plot_baseline_psd(noise_ref, os.path.join(FIGURE_DIR, "fig8_baseline_psd.png"))
    export_baseline_table(noise_ref, os.path.join(TABLE_DIR, "baseline_channels.csv"))

    print("\n  Plotting stage complete.")


if __name__ == "__main__":
    main()
