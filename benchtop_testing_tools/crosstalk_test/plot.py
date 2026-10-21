#!/usr/bin/env python3
"""
Plotting stage of the crosstalk analysis pipeline.
Reads interim data from analyze.py and generates all figures and tables.

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
OUTPUT_DIR = "/home/karl/repos/Sokosti_tools/benchtop_testing_tools/crosstalk_test/outputs"
INTERIM_PATH = os.path.join(OUTPUT_DIR, "interim_data.npz")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

# Constants
FS = 1000
N_CHANNELS = 16
# Shared with standalone_baseline_test/plot.py so both baseline PSD figures
# use the same y-axis (scale and limits) and are directly comparable.
PSD_YLIM = (0.0, 2.5e-2)
PSD_SMOOTH_BINS = 21  # moving-average window (~10 Hz at 0.49 Hz resolution)
SINE_F0 = [10, 100]
AMPLITUDES_MVPP = [2, 10]
CHIP_OF = {ch: ("ads1299-1" if 1 <= ch <= 8 else "ads1299-2") for ch in range(1, 17)}

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

    n_conditions = len(d["matrices_n_f0"])
    crosstalk = {}
    for i in range(n_conditions):
        f0 = int(d["matrices_n_f0"][i])
        amp = int(d["matrices_n_amp"][i])
        key = (f0, amp)
        crosstalk[key] = {
            "driven_amps_uV": {int(k): float(v) for k, v in d["matrices_driven_amps"][i].items()},
            "detection_limits_db": {int(k): float(v) for k, v in d["matrices_detection_limits"][i].items()},
            "xt_db": d["matrices_xt_db"][i].astype(float),
            "xt_corrected": d["matrices_xt_corr"][i].astype(float),
            "censored": d["matrices_censored"][i].astype(bool),
        }

    n_noise = len(d["noise_ch"])
    noise_ref = {}
    for i in range(n_noise):
        ch = int(d["noise_ch"][i])
        noise_ref[ch] = {
            "fft_amps_uV": {10: float(d["noise_fft10"][i]), 100: float(d["noise_fft100"][i])},
            "welch_freqs_hz": d["noise_welch_freqs"][i],
            "welch_psd_uv2": d["noise_welch_psd_uv2"][i],
            "mean_uV": float(d["noise_mean_uV"][i]),
            "rms_uV": float(d["noise_rms_uV"][i]),
            "vpp_uV": float(d["noise_vpp_uV"][i]),
        }

    driven_psds = {}
    for i, key_str in enumerate(d["psd_keys"]):
        f0, amp, driven_ch, ch = key_str.split("_")
        driven_psds[(int(f0), int(amp), int(driven_ch), int(ch))] = (
            d["psd_freqs"][i], d["psd_density_nv"][i])

    baseline_info = d["baseline_info"].item()
    if not isinstance(baseline_info, dict):
        baseline_info = dict(baseline_info)

    return crosstalk, noise_ref, baseline_info, driven_psds


# ============================================================================
# FIGURE FUNCTIONS
# ============================================================================

def plot_parameter_overview(crosstalk, baseline_duration_s, output_path):
    fig, ax = plt.subplots(figsize=_fig_size(7.0))
    ax.axis("off")
    sys_rows = [
        ["Parameter", "Value"],
        ["Sampling rate fs", f"{FS} Hz"], ["PGA gain", "1G"],
        ["Vref", "4.5 V"],
        ["LSB formula", "LSB = 2*Vref / (gain * 2^24)"],
        ["LSB value", f"{2.0*4.5/(1*2**24):.4e} V = {2.0*4.5/(1*2**24)*1e6:.4f} uV/count"],
        ["Channel count", "16"],
        ["Channel mapping", "ch 1-8 -> ads1299-1, ch 9-16 -> ads1299-2"],
        ["Segment duration", "3.0 s"],
        ["File format", "PC.csv (16 EMG channels + sample/status)"],
        ["Baseline duration", f"{baseline_duration_s:.2f} s"],
    ]
    test_rows = [["Condition", "Files"]]
    for (f0, amp) in sorted(crosstalk.keys()):
        test_rows.append([f"16 driven channels, {f0} Hz, {amp} mVpp", "16"])
    table_data = sys_rows + [["", ""]] + test_rows
    table = ax.table(cellText=table_data, loc="upper left", cellLoc="left",
                     colWidths=[0.35, 0.25, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(len(table_data[0])):
        table[1, j].set_facecolor("#D9E2F3")
        table[1, j].set_text_props(fontweight="bold")
    ax.set_title("System Parameters and Test Conditions", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # CSV export of the same table
    csv_path = os.path.join(TABLE_DIR, "parameter_overview.csv")
    with open(csv_path, "w") as f:
        for row in table_data:
            f.write(",".join(row) + "\n")
    print(f"  Saved: {csv_path}")


def plot_psd_overlay(noise_ref, driven_psds, output_dir):
    """PSD overlay per driven channel: driven channel, adjacent victim, far victim
    (all from the same crosstalk recording), and the baseline PSD of the driven
    channel. Log x and log y axes. PSD shown in mV²/Hz. One figure per condition."""
    psd_channels = [1, 4, 8, 9, 13, 16]
    conditions = [(10, 10), (10, 2), (100, 10), (100, 2)]

    for f0, amp in conditions:
        fig, axes = plt.subplots(2, 3, figsize=_fig_size(7.5, 0.75))
        axes = axes.flatten()
        for idx, driven_ch in enumerate(psd_channels):
            ax = axes[idx]
            adj_ch = driven_ch + 1 if driven_ch < 16 else driven_ch - 1
            far_ch = 16 if CHIP_OF[driven_ch] == "ads1299-1" else 1
            series = [
                (driven_ch, f"Driven CH{driven_ch}", "red", 1.0),
                (adj_ch, f"Adjacent CH{adj_ch}", "blue", 1.0),
                (far_ch, f"Far CH{far_ch}", "green", 1.0),
            ]
            for ch, label, color, alpha in series:
                key = (f0, amp, driven_ch, ch)
                if key not in driven_psds:
                    continue
                f_psd, dens = driven_psds[key]
                # dens is in uV²/Hz; /1e6 -> mV²/Hz
                ax.plot(f_psd, dens / 1e6, color=color, linewidth=0.8,
                        alpha=alpha, label=label)
            if driven_ch in noise_ref:
                nr = noise_ref[driven_ch]
                ax.plot(nr["welch_freqs_hz"], nr["welch_psd_uv2"] / 1e6, color="gray",
                        linewidth=0.8, alpha=0.9, label=f"Baseline CH{driven_ch}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

            if f0 == 100:
                ax.set_xlim([50, 500])
                ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
                ax.set_xticks([50, 60, 70, 80, 90, 100, 200, 300, 400, 500], minor=True)
                ax.set_xticklabels(["50", "", "", "", "", "", "", "", "", "500"], minor=True)
                ax.tick_params(axis="x", which="minor")

            else:
                ax.set_xlim([0.5, 500])
                
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (mV²/Hz)")
            ax.set_title(f"CH{driven_ch} driven, {f0} Hz, {amp} mVpp", fontsize=8)
            
            if f0 == 100:
                ax.legend(fontsize=5.5, loc="upper right")
            else:
                ax.legend(fontsize=5.5, loc="upper right")
            
            ax.grid(alpha=0.3, linewidth=0.3, which="both")
        plt.suptitle(f"PSD Overlay – {f0} Hz, {amp} mVpp (log-log)",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"fig2_psd_overlay_{f0}Hz_{amp}mVpp.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {output_path}")


def plot_crosstalk_matrix(crosstalk, output_path):
    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]
    all_vals = []
    for key in conditions:
        if key in crosstalk:
            vals = crosstalk[key]["xt_corrected"]
            mask = ~crosstalk[key]["censored"]
            all_vals.extend(vals[mask].flatten())
    vmin = min(np.min(all_vals), -60) if all_vals else -60
    vmax = max(np.max(all_vals), 0) if all_vals else 0

    # Using layout="constrained" to prevent colorbar/title overlaps
    fig, axes = plt.subplots(2, 2, figsize=_fig_size(7.0, 0.85), layout="constrained")
    axes = axes.flatten()
    
    for idx, (f0, amp) in enumerate(conditions):
        ax = axes[idx]
        if (f0, amp) not in crosstalk:
            continue
        res = crosstalk[(f0, amp)]
        xt = res["xt_corrected"]
        censored = res["censored"]
        
        im = ax.imshow(xt, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
        diag = np.eye(16, dtype=bool)
        ax.imshow(np.where(diag, 1, 0), cmap="gray", alpha=0.3, vmin=0, vmax=1, aspect="auto")
        
        # Draw sensor/hatch marks
        for i in range(16):
            for j in range(16):
                if censored[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, edgecolor="black", hatch="//", linewidth=0.5))
        
        # --- MODIFIED: Write value in EVERY cell (black and larger) ---
        for i in range(16):
            for j in range(16):
                if not censored[i, j] and np.isfinite(xt[i, j]):
                    # Changed to color="black", fontsize=7.5 (up from 6)
                    # Note: Using :.0f (no decimals) so larger text fits inside the tiny cells
                    ax.text(j, i, f"{xt[i, j]:.0f}", 
                            ha="center", va="center", 
                            fontsize=6, color="black")
        
        # Axis labels and ticks
        ax.set_xlabel("Victim channel")
        ax.set_ylabel("Driven channel")
        ax.set_xticks(range(16))
        ax.set_xticklabels([str(c) for c in range(1, 17)], fontsize=6)
        ax.set_yticks(range(16))
        ax.set_yticklabels([str(c) for c in range(1, 17)], fontsize=6)
        dl = np.nanmedian(list(res["detection_limits_db"].values()))
                # 2. Build a 2-line title if DL is valid
        title_text = f"{f0} Hz, {amp} mVpp – XT (dB, corrected)"
        if np.isfinite(dl):
            title_text += f"\n(DL median: {dl:.1f} dB)" 
        ax.set_title(title_text, fontsize=8, pad=8) # Added pad to give space below title

    plt.suptitle("Crosstalk Matrix (noise-corrected, hatched = censored)", fontsize=10, fontweight="bold")
    
    if all_vals:
        cbar = fig.colorbar(im, ax=axes, orientation="vertical", shrink=0.8)
        cbar.set_label("XT (dB)", fontsize=8)
        
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")



def plot_crosstalk_vs_distance(crosstalk, output_path):
    """Mean and max crosstalk per channel distance, one subplot per condition
    (2x2 grid: 4 combinations of frequency and amplitude)."""
    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]
    fig, axes = plt.subplots(2, 2, figsize=_fig_size(7.0, 0.75))
    axes = axes.flatten()
    for idx, (f0, amp) in enumerate(conditions):
        ax = axes[idx]
        key = (f0, amp)
        if key not in crosstalk:
            continue
        res = crosstalk[key]
        xt = res["xt_corrected"]
        censored = res["censored"]
        mean_by_d, max_by_d = {d: [] for d in range(1, 16)}, {d: [] for d in range(1, 16)}
        for i in range(16):
            for j in range(16):
                if i == j or censored[i, j] or not np.isfinite(xt[i, j]):
                    continue
                d = abs((i + 1) - (j + 1))
                mean_by_d[d].append(xt[i, j])
                max_by_d[d].append(xt[i, j])
        distances = sorted(d for d in range(1, 16) if mean_by_d[d])
        means = [np.mean(mean_by_d[d]) for d in distances]
        maxes = [np.max(max_by_d[d]) for d in distances]
        ax.plot(distances, means, "o-", color="#4472C4", label="Mean", linewidth=1.2)
        ax.plot(distances, maxes, "s--", color="#ED7D31", label="Max (worst pair)", linewidth=1.0)
        dl = np.nanmedian(list(res["detection_limits_db"].values()))
        if np.isfinite(dl):
            ax.axhline(dl, linestyle=":", color="gray", alpha=0.7, label="Detection limit")
        ax.set_xlabel("Channel distance")
        ax.set_ylabel("Crosstalk (dB)")
        ax.set_title(f"{f0} Hz, {amp} mVpp", fontsize=9)
        ax.set_xticks(range(1, 16))
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(alpha=0.3, linewidth=0.3)
    plt.suptitle("Crosstalk vs Channel Distance", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_freq_amp_dependence(crosstalk, output_path):
    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]
    cond_means = {}
    for key in conditions:
        if key not in crosstalk:
            continue
        res = crosstalk[key]
        vals = res["xt_corrected"][~res["censored"]]
        vals = vals[np.isfinite(vals)]
        cond_means[key] = np.mean(vals) if len(vals) > 0 else np.nan

    fig, ax = plt.subplots(figsize=_fig_size(3, 2))

    x = np.arange(2)
    width = 0.3
    for k, amp in enumerate(AMPLITUDES_MVPP):
        y = [cond_means.get((10, amp), np.nan), cond_means.get((100, amp), np.nan)]
        bars = ax.bar(x + (k - 0.5) * width, y, width, label=f"{amp} mVpp")
        for xi, yi in zip(x + (k - 0.5) * width, y):
            if np.isfinite(yi):
                ax.text(xi, yi, f"{yi:.1f}", ha="center", va="bottom" if yi < 0 else "top", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["10 Hz", "100 Hz"])
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Mean crosstalk (dB)")
    ax.set_title("Frequency dependence", fontsize=9)
    ax.legend(fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.5)

    plt.suptitle("Frequency and Amplitude Dependence", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def export_summary_table(crosstalk, output_path):
    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]
    rows = []
    for f0, amp in conditions:
        key = (f0, amp)
        if key not in crosstalk:
            continue
        res = crosstalk[key]
        xt = res["xt_corrected"]
        cens = res["censored"]
        off_diag = ~np.eye(16, dtype=bool)
        vals = xt[off_diag & ~cens]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        worst_idx = np.unravel_index(np.nanargmax(np.where(off_diag, xt, np.nan)), xt.shape)
        worst_pair = f"CH{worst_idx[0]+1}->CH{worst_idx[1]+1}"
        n_censored = int((cens & off_diag).sum())
        n_total = 16 * 15
        adj_vals, non_adj_vals = [], []
        for i in range(16):
            for j in range(16):
                if i == j or cens[i, j]:
                    continue
                d = abs((i + 1) - (j + 1))
                if d == 1:
                    adj_vals.append(xt[i, j])
                else:
                    non_adj_vals.append(xt[i, j])
        rows.append({
            "Freq (Hz)": f0, "Amplitude (mVpp)": amp, "Worst pair": worst_pair,
            "Worst XT (dB)": round(float(vals.max()), 2), "Median XT (dB)": round(float(np.median(vals)), 2),
            "Mean adj (dB)": round(float(np.mean(adj_vals)), 2) if adj_vals else "N/A",
            "Mean non-adj (dB)": round(float(np.mean(non_adj_vals)), 2) if non_adj_vals else "N/A",
            "Censored pairs": f"{n_censored}/{n_total}",
            "Detection limit (dB)": round(float(np.nanmedian(list(res["detection_limits_db"].values()))), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_DIR, "summary_crosstalk.csv"), index=False)
    print(f"  Saved: {os.path.join(TABLE_DIR, 'summary_crosstalk.csv')}")
    # LaTeX (booktabs) export
    latex = df.to_latex(index=False, escape=False, column_format="l" * len(df.columns))
    latex = latex.replace("\\toprule", "\\toprule\n").replace("\\midrule", "\\midrule\n").replace("\\bottomrule", "\\bottomrule\n")
    with open(os.path.join(TABLE_DIR, "summary_crosstalk.tex"), "w") as f:
        f.write(latex)
    print(f"  Saved: {os.path.join(TABLE_DIR, 'summary_crosstalk.tex')}")
    return df


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
    ax.set_title("Per-Channel Noise Floor (Baseline)", fontsize=10, fontweight="bold")
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
    along frequency. Y-scale and y-limits match the standalone tool's
    baseline PSD figure (PSD_YLIM)."""
    fig, ax = plt.subplots(figsize=_fig_size(7.0, 0.6))
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    freqs = None
    densities = []
    for ch in range(1, 17):
        nr = noise_ref[ch]
        freqs = nr["welch_freqs_hz"]
        densities.append(np.asarray(nr["welch_psd_uv2"], dtype=float))
        ax.plot(freqs, densities[-1], color=colors[ch - 1], label=f"CH{ch}", linewidth=0.7)
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
    ax.set_title("Baseline PSD – All 16 Channels", fontsize=10, fontweight="bold")
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
    ax.set_title("Baseline Noise per Channel", fontsize=10, fontweight="bold", pad=50)
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


def plot_snr_helper(noise_ref, crosstalk, output_path):
    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]
    fig, axes = plt.subplots(2, 2, figsize=_fig_size(7.0, 0.8))
    axes = axes.flatten()
    for idx, (f0, amp) in enumerate(conditions):
        ax = axes[idx]
        if (f0, amp) not in crosstalk:
            continue
        res = crosstalk[(f0, amp)]
        snr_vals = []
        for ch in range(1, 17):
            A_sig = res["driven_amps_uV"].get(ch, np.nan)
            noise_rms = noise_ref[ch]["rms_uV"]
            if np.isfinite(A_sig) and np.isfinite(noise_rms) and noise_rms > 0:
                snr = 20 * np.log10(A_sig / noise_rms)
            else:
                snr = np.nan
            snr_vals.append(snr)
        ax.bar(range(1, 17), snr_vals, color="#4472C4")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Channel")
        ax.set_ylabel("SNR (dB)")
        ax.set_title(f"{f0} Hz, {amp} mVpp – SNR (RMS: 0-500 Hz)", fontsize=8)
        ax.set_xticks(range(1, 17))
    plt.suptitle("SNR Helper: 20*log10(A_sig / noise_RMS_0-500Hz)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Plotting Stage: Generating Figures and Tables")
    print("=" * 70)

    crosstalk, noise_ref, baseline_info, driven_psds = load_interim()
    print(f"\nLoaded: {len(crosstalk)} conditions, {len(noise_ref)} channels")
    baseline_duration = baseline_info.get("duration_s", 0)

    print("\nGenerating figures and tables...")
    plot_parameter_overview(crosstalk, baseline_duration,
                            os.path.join(FIGURE_DIR, "fig1_parameter_overview.png"))
    plot_psd_overlay(noise_ref, driven_psds,
                     FIGURE_DIR)
    plot_crosstalk_matrix(crosstalk,
                          os.path.join(FIGURE_DIR, "fig3_crosstalk_matrix.png"))
    plot_crosstalk_vs_distance(crosstalk,
                               os.path.join(FIGURE_DIR, "fig4_crosstalk_vs_distance.png"))
    plot_freq_amp_dependence(crosstalk,
                              os.path.join(FIGURE_DIR, "fig5_freq_amp_dependence.png"))
    export_summary_table(crosstalk,
                         os.path.join(TABLE_DIR, "summary_crosstalk.csv"))  # CSV + LaTeX only, no figure
    plot_baseline_noise(noise_ref,
                        os.path.join(FIGURE_DIR, "fig7_baseline_noise.png"))
    plot_baseline_psd(noise_ref,
                      os.path.join(FIGURE_DIR, "fig8_baseline_psd.png"))
    export_baseline_table(noise_ref,
                          os.path.join(TABLE_DIR, "baseline_channels.csv"))
    plot_snr_helper(noise_ref, crosstalk, os.path.join(FIGURE_DIR, "fig10_snr_helper.png"))

    print("\nPlotting stage complete.")


if __name__ == "__main__":
    main()
