#!/usr/bin/env python3
"""
Sit-to-stand GROUP overlay plotter: one figure comparing BLE and USB
directly, built from the same per-trial pipeline CSVs as
plot_sit_stand_trials.py:

    <stem>_signals.csv  - 50 Hz IMU/PCA grid + '# key: value' header comments
    <stem>_emg.csv      - 980 Hz EMG envelope (µV) + header comments

The raw 16-channel EMG is read straight from the original capture CSV
(the '# input:' header comment) for the cycle windows only -- no
intermediate raw-EMG file is written.

Figure layout: two columns (Subject 0 left, Subject 1 right), each overlaying the
USB and BLE gain-1 PC recordings of one subject:

    top (16 mini panels):  raw EMG ch1-ch16 (mV), ALL cycles of both
                           connections overlaid (individual traces)
    below (2 panels):      dynamic accel PC1 (g)   mean + 90 % CI
                           angular-rate PC1 (deg/s) mean + 90 % CI

Colour = connection everywhere (BLE tab:blue, USB tab:orange). The
panels are deliberately "open": no top/right spines, each panel keeps
its own y axis with one uniform y-text size, and the whole figure shares
ONE x axis -- the phase of the sit-to-stand cycle in percent -- whose
tick marks and labels appear ONLY on the bottom row (one percentage
label under the figure), so the timing dependency between the panels
stays visible. All panels live in a single gridspec (one column per
subject), so the raw block and the signal block line up exactly.

Also writes group_overlay_plot_data.csv with exactly the plotted
mean/CI numbers (trial, signal, cycle %, mean, CI low/high, n_cycles)
for the two mean/CI signals; the raw-EMG individual cycles are not
summarised in the CSV.

Usage:
    python plot_sit_stand_group_overlay.py                 # default: PC, gain 1
    python plot_sit_stand_group_overlay.py --out-dir DIR   # custom output folder
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    if os.environ.get("DISPLAY", "") == "" and "--no-plot" not in sys.argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from plot_sit_stand_trials import (
    BASE,
    SIGNALS,
    cycle_curves,
    find_trials,
    load_trial,
    parse_transition_times,
    read_header_comments,
)

SUBJECTS = ("Karl", "Max")
# Display labels: Karl -> Subject 1, Max -> Subject 2 (order kept by SUBJECTS)
SUBJECT_LABELS = {"Karl": "Subject 1", "Max": "Subject 2"}
CONNECTIONS = ("BLE", "USB")
CONNECTION_COLORS = {"BLE": "tab:blue", "USB": "tab:orange"}

# The EMG ch1 envelope cycle-overlay panel is intentionally not plotted;
# the envelope is described in the Methods section instead.
SIGNALS = [s for s in SIGNALS if s[0] != "emg_envelope_uV"]

N_RAW_CHANNELS = 16
# fallback only; the real scale comes from the _emg.csv header
ADS1299_VREF = 4.5

CYCLE_XLABEL = "% of sit-to-stand cycle (0 = start of standing up)"

# ===== FONT SIZES (tune all at once here) =====
# y-axis text (tick numbers + y labels) uses ONE size on every panel
AXIS_TEXT_FSIZE = 15
# x-axis tick numbers on the bottom row (shared % of cycle axis)
XTICK_FSIZE = 15
# figure-wide shared x label under the whole figure
SUPXLABEL_FSIZE = 15
# panel titles (e.g. "Subject 0: raw EMG ch1-ch16 (gain 1)")
RAW_TITLE_FSIZE = 15
# small "No data" placeholder text inside empty panels
NO_DATA_FSIZE = 9
# legend text inside the raw channel and signal panels
LEGEND_FSIZE = 15
# figure-wide top title (suptitle)
SUPTITLE_FSIZE = 20


def find_subject_trials(base, source="PC", gain=1):
    """The gain-1 trials, keyed (subject, connection); rep 1 == gain 1,
    PC source only, '# gain' matching in the EMG header.

    Missing (subject, connection) combinations are simply absent."""
    trials = find_trials(base, source)
    by_key = {}
    for trial_id, signals_path, emg_path in trials:
        m = trial_id.split("_")
        # BLE_1_Karl_Sit_Stand_2026-09-08_PC
        connection, rep, subject = m[0], m[1], m[2]
        if rep != str(gain):
            continue
        emg_meta = read_header_comments(emg_path)
        if emg_meta.get("gain") not in (str(gain), None):
            continue
        if subject not in SUBJECTS or connection not in CONNECTIONS:
            continue
        by_key[(subject, connection)] = (trial_id, signals_path, emg_path)
    return by_key


def compute_curves(meta, signals, emg, cycles, columns):
    """Cycle-locked curves for the requested columns. Returns
    {column: (grid, curves)}; IMU columns are shifted by imu_t0_s into the
    EMG sample-time base, exactly like plot_sit_stand_group.py."""
    t_imu0 = float(meta.get("imu_t0_s", "0"))
    out = {}
    for col in columns:
        if col == "emg_envelope_uV":
            t, y = emg["t_s"].values, emg[col].values
        else:
            t, y = signals["t_s"].values + t_imu0, signals[col].values
        out[col] = cycle_curves(t, y, cycles)
    return out


def mean_and_ci(curves):
    """Mean curve, 90 % CI half-width, and cycle count."""
    n = curves.shape[0]
    mean = curves.mean(axis=0)
    ci = (1.645 * curves.std(axis=0, ddof=1) / np.sqrt(n)
          if n > 1 else np.zeros_like(mean))
    return mean, ci, n


def _open(ax, keep_bottom=False):
    """Open panel style (reference example): all spines removed so the
    panels have no box; optionally keep the bottom spine (the x-axis
    line) for the lowest panel."""
    for s in ("top", "bottom", "left", "right"):
        if s == "bottom" and keep_bottom:
            ax.spines[s].set_visible(True)
        else:
            ax.spines[s].set_visible(False)


def load_raw_emg(meta, emg_path, cycles):
    """Raw ch1-ch16 in mV over the cycle time span, read directly from the
    capture CSV named by the '# input:' header comment (no intermediate
    file). Returns {"t", "Y" (16, N), "gap_rows", "cycles"} or None when
    the capture is missing/unreadable.

    The raw channels carry a large DC offset, so the values are reported
    in mV (the envelope panels keep µV). ``gap_rows`` holds the index of
    the first row after each sample-counter gap so the plotter can break
    the lines there instead of drawing across missing data."""
    capture = meta.get("input")
    emg_fs = float(meta.get("emg_fs_hz", "980"))
    emg_meta = read_header_comments(emg_path)
    if "uv_per_code" in emg_meta:
        uv_per_code = float(emg_meta["uv_per_code"])
    else:
        uv_per_code = ADS1299_VREF / float(emg_meta.get("gain", "8")) / (2 ** 23) * 1e6
    if not capture or not os.path.exists(capture):
        print(f"WARNING: raw EMG capture not found: {capture!r}")
        return None

    ch_cols = [f"ch{i}" for i in range(1, N_RAW_CHANNELS + 1)]
    wanted = ["sample"] + ch_cols
    try:
        df = pd.read_csv(capture, usecols=lambda c: c in wanted)
    except Exception as exc:  # unreadable capture -> skip raw panel only
        print(f"WARNING: reading raw EMG from {capture} failed: {exc}")
        return None
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(f"WARNING: {capture}: missing raw columns {missing}")
        return None

    sample_all = df["sample"].to_numpy(np.int64)
    t_all = sample_all / emg_fs
    lo_t = min(a for a, _ in cycles)
    hi_t = max(b for _, b in cycles)
    inside = (t_all >= lo_t) & (t_all <= hi_t)
    if not inside.any():
        print(f"WARNING: {capture}: no raw EMG samples inside the cycles")
        return None
    idx = np.flatnonzero(inside)
    lo, hi = int(idx[0]), int(idx[-1]) + 1

    sample = sample_all[lo:hi]
    gap_rows = np.flatnonzero(np.diff(sample) != 1) + 1
    # codes -> µV (uv_per_code) -> mV, because of the large DC offset
    Y = df[ch_cols].to_numpy(np.float64)[lo:hi] * uv_per_code / 1000.0
    return {"t": sample / emg_fs, "Y": Y, "gap_rows": gap_rows,
            "cycles": cycles}


def draw_raw_cell(axes, raw_by_conn, title, is_left):
    """Raw EMG ch1-ch16: every cycle of every connection overlaid as
    individual traces, colour = connection, no means/CIs. One open axes
    per channel with its own y axis (mV). y labels on the LEFT column
    only; the RIGHT column shows the tick numbers but no unit labels.
    Legends are NOT drawn here - the connection legend is consolidated
    once in the signal block."""
    for i, ax in enumerate(axes):
        n_plotted = 0
        for conn in CONNECTIONS:
            data = raw_by_conn.get(conn)
            if not data:
                continue
            t, Y = data["t"], data["Y"]
            gap_rows, cycles = data["gap_rows"], data["cycles"]
            color = CONNECTION_COLORS[conn]
            for k, (a, b) in enumerate(cycles):
                m = (t >= a) & (t <= b)
                if m.sum() < 10:
                    continue
                mp = np.flatnonzero(m)
                x = (t[m] - a) / (b - a) * 100.0
                y = Y[m, i]
                if gap_rows.size:
                    # break the line at packet gaps: insert a NaN pair
                    # between the pre- and post-gap sample
                    pos = np.searchsorted(mp, gap_rows)
                    keep = ((pos < mp.size)
                            & (mp[np.clip(pos, 0, mp.size - 1)] == gap_rows))
                    pos = np.unique(pos[keep])
                    if pos.size:
                        x = np.insert(x, pos, np.nan)
                        y = np.insert(y, pos, np.nan)
                n_plotted += 1
                ax.plot(x, y, color=color, lw=0.4, alpha=0.55)
        if n_plotted == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=NO_DATA_FSIZE)
        if is_left:
            ax.set_ylabel(f"ch{i + 1} (mV)", fontsize=AXIS_TEXT_FSIZE)
            ax.tick_params(axis="y", labelsize=AXIS_TEXT_FSIZE, left=False)
        else:
            # numbers only: keep tick labels, drop the unit y-label and
            # the tick marks (open-panel style)
            ax.tick_params(axis="y", labelsize=AXIS_TEXT_FSIZE, left=False)
        ax.tick_params(bottom=False)
        _open(ax)
        ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5)
    axes[0].set_title(title, fontsize=RAW_TITLE_FSIZE)


def draw_signal_cell(axes, curves_by_conn, is_left):
    """Stacked subplots (accel PC1, rate PC1); the USB and BLE means +
    90 % CI are overlaid in each, colour = connection. Open panels (no
    top/right spines); the x axis is shared figure-wide, so no per-panel
    x label here. y unit labels on the LEFT column only; the RIGHT
    column shows the tick numbers but no unit labels. ONE legend per
    subject, drawn once on the top-left signal panel (k == 0), covering
    all four signal panels."""
    for k, (ax, (col, label, unit)) in enumerate(zip(axes, SIGNALS)):
        n_plotted = 0
        for conn in CONNECTIONS:
            data = curves_by_conn.get(conn)
            if not data:
                continue
            grid, curves = data[col]
            if curves is None:
                continue
            n_plotted += 1
            mean, ci, n = mean_and_ci(curves)
            color = CONNECTION_COLORS[conn]
            if ci.any():
                ax.fill_between(grid, mean - ci, mean + ci, color=color,
                                alpha=0.25, lw=0,
                                label=f"{conn} 90% CI" if (is_left and k == 0)
                                      else None)
            ax.plot(grid, mean, color=color, lw=1.3,
                    label=f"{conn} mean (n={n})" if (is_left and k == 0)
                    else None)
        if n_plotted == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=NO_DATA_FSIZE)
        if is_left:
            ax.set_ylabel(f"{label} ({unit})", fontsize=AXIS_TEXT_FSIZE)
            ax.tick_params(axis="y", labelsize=AXIS_TEXT_FSIZE, left=False)
        else:
            # numbers only: keep tick labels, drop the unit y-label and
            # the tick marks (open-panel style)
            ax.tick_params(axis="y", labelsize=AXIS_TEXT_FSIZE, left=False)
        if ax is not axes[-1]:
            ax.tick_params(bottom=False)
        _open(ax, keep_bottom=(ax is axes[-1]))
        ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5)

def make_overlay_figure(subject_curves, subject_raw, gain, suptitle,
                        out_path):
    """One figure, two columns (Subject 0 left, Subject 1 right): 16 raw-EMG
    channel panels on top, a spacer row, then accel PC1 / ang-rate PC1. ALL
    panels come from one single gridspec (one column per subject), so
    the raw block and the signal block are exactly aligned
    horizontally. Every panel has its own y axis, all panels share ONE
    x axis (% of cycle) whose tick marks and labels appear only on the
    bottom row, and the spines stay open."""
    fig = plt.figure(figsize=(16, 24), constrained_layout=True)
    fig.suptitle(suptitle, fontsize=SUPTITLE_FSIZE)
    # one row per raw channel, a spacer row, then the signal panels,
    # with a short gap row between accel PC1 and ang. rate PC1.
    # hspace=0 makes the panels touch so vertical grid lines run
    # continuously (reference example style).  The spacer row is
    # deliberately tall to give a clear visual gap between the raw
    # EMG block and the signal block.
    n_signals = len(SIGNALS)
    # heights: legend row, raw block, spacer, then each signal panel
    # followed by a gap row (except after the last signal). The legend
    # row sits between the suptitle and the raw plots so the figure
    # legend has its own band and never overlaps the panels.
    heights = [0.55] + [1] * N_RAW_CHANNELS + [0.1]
    for k in range(n_signals):
        heights.append(1.67)
        if k < n_signals - 1:
            heights.append(0.3)
    outer = fig.add_gridspec(len(heights), 2,
                             height_ratios=heights, hspace=0)
    anchor = None
    all_axes = []
    bottom_axes = []
    for col, subject in enumerate(SUBJECTS):
        is_left = (col == 0)
        curves_by_conn = subject_curves.get(subject, {})
        raw_by_conn = subject_raw.get(subject, {})

        raw_axes = []
        for r in range(N_RAW_CHANNELS):
            ax = (fig.add_subplot(outer[1 + r, col]) if anchor is None
                  else fig.add_subplot(outer[1 + r, col], sharex=anchor))
            if anchor is None:
                anchor = ax
            raw_axes.append(ax)
        draw_raw_cell(raw_axes, raw_by_conn,
                      f"{SUBJECT_LABELS[subject]}: raw EMG ch1-ch16 "
                      f"(gain {gain})", is_left)

        sig_axes = []
        # signal k sits after the legend, raw block and spacer, skipping
        # over the gap row that follows each previous signal
        for k in range(n_signals):
            row = N_RAW_CHANNELS + 2 + k * 2
            ax = fig.add_subplot(outer[row, col], sharex=anchor)
            sig_axes.append(ax)
        draw_signal_cell(sig_axes, curves_by_conn, is_left)

        all_axes += raw_axes + sig_axes
        bottom_axes.append(sig_axes[-1])

    # One common x axis: tick marks and labels only on the bottom row,
    # one shared percentage label underneath the whole figure.
    anchor.set_xlim(0.0, 100.0)
    for ax in all_axes:
        show = ax in bottom_axes
        ax.tick_params(axis="x", bottom=show, labelbottom=show,
                       labelsize=XTICK_FSIZE)
    fig.supxlabel(CYCLE_XLABEL, fontsize=SUPXLABEL_FSIZE)

    # ONE figure-wide legend above the panels, in its own grid row
    # between the suptitle and the raw plots (applies to all subplots)
    leg_ax = fig.add_subplot(outer[0, :])
    leg_ax.axis("off")
    handles, labels = [], []
    for ax in all_axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    # dedupe by label, keep first occurrence
    seen, uniq = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq.append((h, l))
    leg_ax.legend([h for h, _ in uniq], [l for _, l in uniq],
                  loc="center", frameon=False, ncol=len(CONNECTIONS) * 2,
                  fontsize=LEGEND_FSIZE)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.splitext(out_path)[0] + f".{ext}", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE,
                    help="Root folder to search for the trial CSVs")
    ap.add_argument("--out-dir",
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="Folder for the figure and the plot-data CSV")
    ap.add_argument("--source", choices=("PC", "SD"), default="PC",
                    help="Recording source (default: PC)")
    ap.add_argument("--gain", type=int, default=1,
                    help="Recording gain (default: 1)")
    ap.add_argument("--no-plot", action="store_true",
                    help="Do not open an interactive plot window")
    args = ap.parse_args(argv)

    if plt is None:
        raise SystemExit("matplotlib is required for the plotter")

    group = find_subject_trials(args.base, args.source, args.gain)
    if not group:
        raise SystemExit(f"No matching gain-{args.gain} {args.source} "
                         f"Sit_Stand trials found under {args.base}")
    print(f"Found {len(group)} trials: "
          + ", ".join(t for t, _, _ in group.values()))

    subject_curves = {}
    subject_raw = {}
    plot_rows = []
    for (subject, connection), (trial_id, signals_path,
                                emg_path) in group.items():
        loaded = load_trial(signals_path, emg_path)
        if loaded is None:
            print(f"{trial_id}: no transition times, skipping")
            continue
        meta, signals, emg = loaded
        times = parse_transition_times(meta)
        cycles = list(zip(times[::2], times[1::2]))
        columns = [c for c, _, _ in SIGNALS]
        curves_by_col = compute_curves(meta, signals, emg, cycles, columns)
        subject_curves.setdefault(subject, {})[connection] = curves_by_col
        raw = load_raw_emg(meta, emg_path, cycles)
        if raw is not None:
            subject_raw.setdefault(subject, {})[connection] = raw
        print(f"{trial_id}: {len(cycles)} cycles"
              + ("" if raw is not None else " (raw EMG unavailable)"))
        for col in columns:
            grid, curves = curves_by_col[col]
            if curves is None:
                continue
            n = curves.shape[0]
            mean, ci, _n = mean_and_ci(curves)
            for x, m, lo, hi in zip(grid, mean, mean - ci, mean + ci):
                plot_rows.append({
                    "trial_id": trial_id,
                    "signal": col,
                    "cycle_pct": x,
                    "mean": m,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_cycles": n,
                })

    for subject in SUBJECTS:
        present = subject_curves.get(subject, {})
        missing = [c for c in CONNECTIONS if c not in present]
        if missing:
            print(f"WARNING: {subject}: missing {', '.join(missing)} data")

    fig_path = os.path.join(args.out_dir, "sit_stand_group_overlay.png")
    suptitle = (f"Sit-to-stand cycle overlays: gain-{args.gain} "
                f"{args.source} recordings - BLE vs USB per subject")
    make_overlay_figure(subject_curves, subject_raw, args.gain, suptitle,
                        fig_path)

    plot_data = pd.DataFrame(plot_rows)
    csv_path = os.path.join(args.out_dir, "group_overlay_plot_data.csv")
    plot_data.to_csv(csv_path, index=False)

    print(f"written: {os.path.splitext(fig_path)[0]}.png/.pdf")
    print(f"written: {csv_path}  ({len(plot_data)} rows)")

    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
