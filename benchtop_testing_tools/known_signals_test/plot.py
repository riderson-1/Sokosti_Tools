#!/usr/bin/env python3
"""
Plotting stage of the known-signal test analysis pipeline.
Reads interim data from analyze.py and generates all figures and tables,
so plots can be changed without re-running the analysis.

Usage:
    python3 plot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths: output goes to an "output" subfolder next to this script (same
# convention as analyze.py), regardless of the directory the script is run
# from.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
INTERIM_PATH = os.path.join(OUTPUT_DIR, "interim_data.npz")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FS = 1000
FIG1_CH = 1  # channel shown in the waveform figure
N_PERIODS = 4  # plotted span per panel (within 3-5 periods)

# ============================================================================
# FONT SIZES (Figure 1 waveforms)
# ============================================================================
# Global rcParams defaults (used by all figures).
FONT_GLOBAL = 9          # base font size
FONT_GLOBAL_TICK = 8     # tick labels
FONT_GLOBAL_LEGEND = 7.5 # legend entries

# Figure 1 (waveform grid) overrides -- increase these to make fig 1 readable.
FIG1_FONT_TITLE = 14     # panel titles ("10 Hz", ...)
FIG1_FONT_YLABEL = 13    # "Amplitude (mV)" / "Residual (µV)"
FIG1_FONT_XLABEL = 13    # "Time (ms)"
FIG1_FONT_ROWLABEL = 13  # row labels ("1 G, 2 mVpp")
FIG1_FONT_TICK = 11      # axis tick numbers
FIG1_FONT_LEGEND = 11    # legend ("Recorded data", "Fitted sine")

# Figure 2 (amplitude ratio) and Figure 3 (SINAD/ENOB) overrides.
FIG23_FONT_TITLE = 14    # figure title ("Amplitude ratio", "SINAD / ENOB")
FIG23_FONT_YLABEL = 13   # y-axis labels
FIG23_FONT_XLABEL = 13   # "Frequency (Hz)"
FIG23_FONT_TICK = 11     # axis tick numbers
FIG23_FONT_LEGEND = 11   # legend entries ("1 G, 2 mVpp", ...)
FIG23_LEGEND_HEIGHT = 22 # space reserved above the axes for the legend (pt)
FIG23_TITLE_PAD = 60     # extra gap between title and legend/axes (pt)

plt.rcParams.update({
    "font.size": FONT_GLOBAL, "axes.labelsize": FONT_GLOBAL,
    "axes.titlesize": FONT_GLOBAL,
    "xtick.labelsize": FONT_GLOBAL_TICK, "ytick.labelsize": FONT_GLOBAL_TICK,
    "legend.fontsize": FONT_GLOBAL_LEGEND,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
})

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def save_fig(fig, path):
    """Save a figure as PNG (300 dpi) and PDF."""
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {os.path.basename(path)} + .pdf")


def load_interim():
    """Load interim data from analyze.py.

    Looks in the local output/ folder first, then falls back to the copy
    saved next to the raw recordings (root folder of the test data).
    """
    candidates = [INTERIM_PATH,
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "interim_data.npz")]
    for path in candidates:
        if os.path.isfile(path):
            d = np.load(path)
            return {k: d[k] for k in d.files}
    raise FileNotFoundError(
        "interim_data.npz not found. Looked in:\n  "
        + "\n  ".join(candidates)
        + "\nRun analyze.py first (it saves a copy next to the raw data).")


def find_condition(d, gain, amp, freq):
    """Index of a (gain, amp, freq) condition in the interim arrays, or None."""
    idx = np.flatnonzero((d["gains"] == gain)
                         & np.isclose(d["amps_mvpp"], amp)
                         & (d["freqs_hz"] == freq))
    return int(idx[0]) if len(idx) else None


def pick_unit(x):
    """Choose a display unit (scale, label) for a value x in volts."""
    x = abs(float(x))
    if x >= 1e-3:
        return 1e3, "mV"
    if x >= 1e-6:
        return 1e6, "µV"
    return 1e9, "nV"


# ============================================================================
# FIGURE 1: data vs. fitted ideal sine, all conditions
# ============================================================================

def plot_waveforms(d, output_path):
    """Full grid: rows = (gain, amplitude), columns = frequency.

    The fitted sine is evaluated analytically on a dense grid (smooth curve,
    drawn in the background); the recorded samples are drawn on top.
    The x-axis uses the measured sample rate (fs_eff), so the period shown
    is the true one (~980 SPS, not the nominal 1000).
    """
    gains = sorted({int(g) for g in d["gains"]})
    amps = sorted({float(a) for a in d["amps_mvpp"]})
    combos = [(g, a) for g in gains for a in amps]
    freqs = sorted({int(f) for f in d["freqs_hz"]})
    n_rows, n_cols = len(combos), len(freqs)

    fig = plt.figure(figsize=(3.3 * n_cols + 1.2, 2.5 * n_rows + 1.0))
    outer = fig.add_gridspec(n_rows, n_cols, wspace=0.18, hspace=0.16)
    outer.update(bottom=0.05, top=0.96, left=0.10, right=0.99)

    for r, (gain, amp) in enumerate(combos):
        for c, freq in enumerate(freqs):
            cell = outer[r, c].subgridspec(
                2, 1, height_ratios=[3, 2], hspace=0.08)
            ax = fig.add_subplot(cell[0])
            axr = fig.add_subplot(cell[1], sharex=ax)
            ax.tick_params(labelbottom=False,
                           labelsize=FIG1_FONT_TICK)
            axr.tick_params(labelsize=FIG1_FONT_TICK)

            idx = find_condition(d, gain, amp, freq)
            if idx is None:
                print(f"  WARNING: {gain}G/{amp}mVpp/{freq}Hz not in interim "
                      f"data, leaving panel empty")
                ax.set_axis_off()
                axr.set_axis_off()
                continue

            fs_eff = float(d["fs_eff_sps"][idx])
            raw_full = d["data_v"][idx, FIG1_CH - 1].astype(float)
            a = float(d["fit_a"][idx, FIG1_CH - 1])
            b = float(d["fit_b"][idx, FIG1_CH - 1])
            c0 = float(d["fit_c"][idx, FIG1_CH - 1])
            resid_full = raw_full - d["fit_v"][idx, FIG1_CH - 1]

            # Span of N_PERIODS at the measured rate
            n_per = fs_eff / freq
            n_show = min(int(round(N_PERIODS * n_per)), len(raw_full))
            x_ms = np.arange(n_show) / fs_eff * 1e3

            # Display-only: remove slow baseline drift (quadratic trend of
            # the residual). The analysis uses the raw data; the fit and its
            # coefficients are untouched.
            tn = np.linspace(-1.0, 1.0, n_show)
            trend = np.polyval(np.polyfit(tn, resid_full[:n_show], 2), tn)
            raw = raw_full[:n_show] - trend
            resid = resid_full[:n_show] - trend

            # Smooth analytical fit on a dense grid (background)
            u = np.linspace(0, n_show - 1, max(600, 8 * n_show))
            w = 2.0 * np.pi * freq * u / fs_eff
            fit_dense = a * np.sin(w) + b * np.cos(w) + c0

            scale, unit = pick_unit(np.max(np.abs(raw)))
            rscale, runit = pick_unit(np.max(np.abs(resid)))

            lbl_data = "Recorded data"
            lbl_fit = "Fitted sine"
            # Fit first (background), recorded samples on top (foreground)
            ax.plot(u / fs_eff * 1e3, fit_dense * scale, lw=1.8, color="C3",
                    zorder=1.5, label=lbl_fit)
            marker = "o" if n_per <= 8 else None
            ax.plot(x_ms, raw * scale, lw=1.0, color="C0",
                    marker=marker, ms=3.0, markevery=2,
                    zorder=2.5, label=lbl_data)

            axr.plot(x_ms, resid * rscale, lw=0.8, color="0.25", zorder=2)
            ax.grid(True, ls=":", lw=0.5, alpha=0.6)
            axr.grid(True, ls=":", lw=0.5, alpha=0.6)

            if r == 0:
                ax.set_title(f"{freq} Hz", fontsize=FIG1_FONT_TITLE)
            if c == 0:
                ax.set_ylabel(f"Amplitude ({unit})", fontsize=FIG1_FONT_YLABEL)
                ax.text(-0.36, 0.5, f"{gain} G, {amp:g} mVpp",
                        transform=ax.transAxes, rotation=90,
                        ha="center", va="center",
                        fontsize=FIG1_FONT_ROWLABEL, fontweight="bold")
            if r == n_rows - 1:
                axr.set_xlabel("Time (ms)", fontsize=FIG1_FONT_XLABEL)
            else:
                axr.tick_params(labelbottom=False)

    # Boxed legend centered at the bottom of the page
    handles = [
        plt.Line2D([], [], color="C0", lw=1.0, marker="o", ms=3.0,
                   label="Recorded data"),
        plt.Line2D([], [], color="C3", lw=1.8, label="Fitted sine"),
        plt.Line2D([], [], color="0.25", lw=0.8, label="Residual"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=True, edgecolor="0.6", fancybox=False,
               fontsize=FIG1_FONT_LEGEND, bbox_to_anchor=(0.5, -0.01))
    save_fig(fig, output_path)


# ============================================================================
# FIGURE 2: amplitude ratio vs. frequency
# ============================================================================

def plot_amplitude_ratio(d, output_path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    combos = sorted({(int(g), float(a))
                     for g, a in zip(d["gains"], d["amps_mvpp"])})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for k, (gain, amp) in enumerate(combos):
        mask = (d["gains"] == gain) & np.isclose(d["amps_mvpp"], amp)
        freqs = d["freqs_hz"][mask]
        order = np.argsort(freqs)
        freqs = freqs[order]
        # Per-channel ratio measured/nominal, mean and spread across channels
        ratios = d["vpp_v"][mask][order] / (amp * 1e-3)
        mean = np.nanmean(ratios, axis=1)
        std = np.nanstd(ratios, axis=1, ddof=1)
        ax.errorbar(freqs, mean, yerr=std, color=colors[k], marker="o",
                    ms=4.5, lw=1.3, capsize=3,
                    label=f"{gain} G, {amp:g} mVpp")

    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="Nominal (1.0)")

    ax.set_xscale("log")
    all_freqs = sorted(set(int(f) for f in d["freqs_hz"]))
    ax.set_xticks(all_freqs)
    ax.set_xticklabels([str(f) for f in all_freqs])
    ax.set_xlabel("Frequency (Hz)", fontsize=FIG23_FONT_XLABEL)
    ax.set_ylabel("Amplitude ratio (measured / nominal)",
                  fontsize=FIG23_FONT_YLABEL)
    ax.set_title("Amplitude ratio vs. frequency",
                 fontsize=FIG23_FONT_TITLE, pad=FIG23_TITLE_PAD)
    ax.tick_params(labelsize=FIG23_FONT_TICK)
    ax.grid(True, which="major", ls=":", lw=0.6, alpha=0.7)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
              frameon=False, fontsize=FIG23_FONT_LEGEND,
              columnspacing=1.2, handlelength=1.6,
              borderaxespad=0)

    save_fig(fig, output_path)


# ============================================================================
# FIGURE 3: SINAD (dB) / ENOB (bits) vs frequency
# ============================================================================

def plot_sinad(d, output_path):
    """SINAD in dB (left axis) with ENOB in bits on the right axis.

    ENOB = (SINAD - 1.76) / 6.02 is a linear re-scaling of the same curve,
    so both units share one plot.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    combos = sorted({(int(g), float(a))
                     for g, a in zip(d["gains"], d["amps_mvpp"])})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for k, (gain, amp) in enumerate(combos):
        mask = (d["gains"] == gain) & np.isclose(d["amps_mvpp"], amp)
        freqs = d["freqs_hz"][mask]
        order = np.argsort(freqs)
        freqs = freqs[order]
        sinad = d["sinad_db"][mask][order]
        mean = np.nanmean(sinad, axis=1)
        std = np.nanstd(sinad, axis=1, ddof=1)
        ax.errorbar(freqs, mean, yerr=std, color=colors[k], marker="o",
                    ms=4.5, lw=1.3, capsize=3,
                    label=f"{gain} G, {amp:g} mVpp")

    ax.axhline(0.0, color="k", ls="--", lw=1.0,
               label="0 dB (tone = residual)")

    ax.set_xscale("log")
    all_freqs = sorted(set(int(f) for f in d["freqs_hz"]))
    ax.set_xticks(all_freqs)
    ax.set_xticklabels([str(f) for f in all_freqs])
    ax.set_xlabel("Frequency (Hz)", fontsize=FIG23_FONT_XLABEL)
    ax.set_ylabel("SINAD (dB)", fontsize=FIG23_FONT_YLABEL)
    ax.set_title("SINAD / ENOB vs. frequency",
                 fontsize=FIG23_FONT_TITLE, pad=FIG23_TITLE_PAD)
    ax.tick_params(labelsize=FIG23_FONT_TICK)
    ax.grid(True, which="major", ls=":", lw=0.6, alpha=0.7)

    # Right axis: ENOB is an affine map of SINAD -> rescale the limits
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    ax2 = ax.twinx()
    ax2.set_ylim((lo - 1.76) / 6.02, (hi - 1.76) / 6.02)
    ax2.set_ylabel("ENOB (bits)", fontsize=FIG23_FONT_YLABEL)
    ax2.tick_params(labelsize=FIG23_FONT_TICK)

    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
              frameon=False, fontsize=FIG23_FONT_LEGEND,
              columnspacing=1.2, handlelength=1.6,
              borderaxespad=0)
    save_fig(fig, output_path)


# ============================================================================
# RESULTS TABLE: compact matrix + LaTeX (booktabs)
# ============================================================================

def export_results_table(d, csv_path, tex_path):
    """Matrix grouped by gain then nominal Vpp, frequency across columns."""
    gains = sorted(set(int(g) for g in d["gains"]))
    amps = sorted(set(float(a) for a in d["amps_mvpp"]))
    freqs = sorted(set(int(f) for f in d["freqs_hz"]))
    index = pd.MultiIndex.from_product([gains, amps], names=["gain", "amp_mvpp"])
    columns = pd.Index(freqs, name="freq_hz")

    err = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    sinad = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    enob = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    for g, a, f, e, s, b in zip(d["gains"], d["amps_mvpp"], d["freqs_hz"],
                                np.nanmean(d["amp_err_pct"], axis=1),
                                np.nanmean(d["sinad_db"], axis=1),
                                np.nanmean(d["enob_bits"], axis=1)):
        err.loc[(int(g), float(a)), int(f)] = e
        sinad.loc[(int(g), float(a)), int(f)] = s
        enob.loc[(int(g), float(a)), int(f)] = b

    # CSV: all metrics, one row per (gain, amplitude)
    out = pd.concat({"amp_err_pct": err, "sinad_db": sinad,
                     "ENOB (FS-normalized)": enob},
                    names=["metric"]).rename_axis(
                        ["metric", "gain", "amp_mvpp"]).reset_index()
    out.to_csv(csv_path, index=False)

    # LaTeX (booktabs): three matrices, missing conditions shown as "--"
    def to_tex(block):
        disp = block.copy()
        disp.columns = [f"{f} Hz" for f in disp.columns]
        disp.index = pd.MultiIndex.from_tuples(
            [(g, f"{a:g}") for g, a in disp.index],
            names=["Gain", r"Nominal $V_\mathrm{pp}$ (mV)"])
        disp = disp.map(lambda v: f"{v:.2f}" if np.isfinite(v) else "--")
        return disp.to_latex(column_format="ll" + "c" * len(freqs))

    with open(tex_path, "w") as f:
        f.write("% Mean amplitude error across channels, in percent\n")
        f.write(to_tex(err))
        f.write("\n\n% Mean SINAD across channels, in dB (primary quality metric)\n")
        f.write(to_tex(sinad))
        f.write("\n\n% Mean ENOB (FS-normalized) across channels, in bits "
                "(MT-003 Eq. 2)\n")
        f.write(to_tex(enob))
        f.write("\n")

    print(f"  {os.path.basename(csv_path)}")
    print(f"  {os.path.basename(tex_path)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Known-signal test: plotting stage")
    print("=" * 70)

    d = load_interim()

    print("\nFigure 1: recorded data vs. fitted sine, all conditions...")
    plot_waveforms(d, os.path.join(FIGURE_DIR, "fig1_waveforms.png"))

    print("Figure 2: amplitude ratio vs. frequency...")
    plot_amplitude_ratio(d, os.path.join(FIGURE_DIR, "fig2_amplitude_ratio.png"))

    print("Figure 3: SINAD / ENOB vs. frequency...")
    plot_sinad(d, os.path.join(FIGURE_DIR, "fig3_sinad.png"))

    print("Results table...")
    export_results_table(d,
                         os.path.join(TABLE_DIR, "results_table.csv"),
                         os.path.join(TABLE_DIR, "results_table.tex"))

    print("\nPlotting stage complete.")


if __name__ == "__main__":
    main()
