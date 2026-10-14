#!/usr/bin/env python3
"""
Gait GROUP overlay plotter: ONE figure comparing the three walk types
over USB and BLE, built from the same per-trial pipeline CSVs as
plot_gait_trials.py (written by gait_analysis.py --save-trial-csv):

    <stem>_signals.csv  - 50 Hz IMU/PCA grid + '# key: value' header comments
    <stem>_emg.csv      - 980 Hz EMG envelope (µV) + header comments

Figure layout: 3 rows x 4 columns, Max only (Karl is not considered).
Rows are the signals; each connection gets an overlay plot with a box
plot of its per-cycle values directly to its right (metric chosen with
--box-metric: rms = default, p2p = peak-to-peak):

    row 1: EMG envelope (µV)
    row 2: dynamic accel PC1 (g)
    row 3: angular-rate PC1 (deg/s)

    col 1: USB overlays      col 2: USB metric boxes
    col 3: BLE overlays      col 4: BLE metric boxes

Every overlay cell draws Slow / Fast / Incline mean + 90 % CI
(Slow tab:blue, Fast tab:green, Incline tab:red). Every box-plot cell
shows one coloured box per walk type with that row's PER-CYCLE values
in the metric selected with --box-metric: rms = sqrt(mean(y^2)) over
the cycle's samples (default), p2p = cycle max - min.

The cycle graphs use open panels (no top/right spines) and share ONE
x axis (% of gait cycle, labelled once under the whole figure); the box
plots keep their categorical walk-type x axis.

Also writes gait_overlay_plot_data.csv (plotted mean/CI curves) and
gait_overlay_box_data.csv (BOTH per-cycle box metrics: one row per
cycle with a metric column, "rms" and "p2p").

Usage:
    python plot_gait_group_overlay.py                   # RMS boxes (default)
    python plot_gait_group_overlay.py --box-metric p2p  # peak-to-peak boxes
    python plot_gait_group_overlay.py --out-dir DIR     # custom output folder
    python plot_gait_group_overlay.py --walk-type Incline
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

from plot_gait_trials import (
    BASE,
    SIGNALS,
    cycle_curves,
    find_trials,
    load_trial,
    parse_heel_strikes,
    read_header_comments,
)

SUBJECT = "Max"
CONNECTIONS = ("USB", "BLE")  # layout: USB left half, BLE right half

WALK_TYPES = ("Slow", "Fast", "Incline")
WALK_COLORS = {"Slow": "tab:blue", "Fast": "tab:green", "Incline": "tab:red"}

CYCLE_XLABEL = "% of gait cycle (0 = heel strike)"

# Box-plot metric (--box-metric): rms = per-cycle RMS of the signal as
# recorded (default), p2p = per-cycle peak-to-peak (max - min).
BOX_METRICS = ("rms", "p2p")
METRIC_LABEL = {"rms": "RMS", "p2p": "peak-to-peak"}


def find_max_trials(base, walk_type, source="PC", gain=1):
    """The gain-1 Max trials for one walk type, keyed by connection
    (USB / BLE); rep 1 == gain 1, PC source only, '# gain' matching in
    the EMG header. Missing connections are simply absent."""
    trials = find_trials(base, source, walk_type)
    by_conn = {}
    for trial_id, signals_path, emg_path in trials:
        m = trial_id.split("_")
        # BLE_1_Max_Fast_Walk_2026-09-08_PC
        connection, rep, subject = m[0], m[1], m[2]
        if rep != str(gain) or subject != SUBJECT:
            continue
        emg_meta = read_header_comments(emg_path)
        if emg_meta.get("gain") not in (str(gain), None):
            continue
        if connection not in CONNECTIONS:
            continue
        by_conn[connection] = (trial_id, signals_path, emg_path)
    return by_conn


def compute_curves(meta, signals, emg, cycles, columns):
    """Gait-cycle-locked curves for the requested columns. Returns
    {column: (grid, curves)}; IMU columns are shifted by imu_t0_s into the
    EMG sample-time base, exactly like the other group plotters."""
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


def _open(ax):
    """Open panel style: keep only the left and bottom spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_overlay(ax, curves_by_walk, label, unit, title):
    """Slow/Fast/Incline means + 90 % CI on one open axes, one colour per
    walk type; no individual cycle traces."""
    n_plotted = 0
    for walk in WALK_TYPES:
        data = curves_by_walk.get(walk)
        if not data:
            continue
        grid, curves = data
        if curves is None:
            continue
        n_plotted += 1
        mean, ci, n = mean_and_ci(curves)
        color = WALK_COLORS[walk]
        if ci.any():
            ax.fill_between(grid, mean - ci, mean + ci, color=color,
                            alpha=0.25, lw=0, label=f"{walk} 90% CI")
        ax.plot(grid, mean, color=color, lw=1.4,
                label=f"{walk} mean (n={n})")
    if n_plotted == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
    else:
        ax.legend(loc="upper right", fontsize=7)
    ax.set_ylabel(f"{label} ({unit})")
    if title:
        ax.set_title(title, fontsize=10)
    _open(ax)
    ax.grid(True, alpha=0.3)


def draw_boxplot(ax, vals_by_walk, metric, unit, title):
    """Box plots for the metric selected with --box-metric: one coloured
    box per walk type; each data point is one gait cycle's value of that
    row's signal (rms = sqrt(mean(y^2)), p2p = max - min)."""
    series, positions, colors, names = [], [], [], []
    for i, walk in enumerate(WALK_TYPES, start=1):
        vals = vals_by_walk.get(walk)
        if vals is None or len(vals) == 0:
            continue
        series.append(np.asarray(vals, dtype=float))
        positions.append(i)
        colors.append(WALK_COLORS[walk])
        names.append(walk)
    if not series:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
    else:
        bp = ax.boxplot(series, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color="black", lw=1.2),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="0.4",
                                        markeredgecolor="0.4"))
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.55)
            box.set_edgecolor(color)
        ax.set_xticks(positions)
        ax.set_xticklabels(names)
        ax.set_xlim(0.5, len(WALK_TYPES) + 0.5)
    ax.set_ylabel(f"{METRIC_LABEL[metric]} ({unit})")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)


def make_overlay_figure(curves, box, metric, suptitle, out_path):
    """One figure, 3 rows (signals) x 4 columns
    [USB overlay | USB box | BLE overlay | BLE box]. The cycle overlays
    share ONE x axis (% of cycle, labelled once under the whole figure)
    and stay open; the box plots show the per-cycle values of `metric`
    (rms or p2p) on their categorical walk-type x axis with labels only
    on the bottom row."""
    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    fig.suptitle(suptitle)
    outer = fig.add_gridspec(3, 4, width_ratios=[1.4, 0.6, 1.4, 0.6])
    anchor = None
    cycle_axes = []
    box_axes = []
    bottom_axes = []
    for row, (col, label, unit) in enumerate(SIGNALS):
        for k, conn in enumerate(CONNECTIONS):
            plot_col = 2 * k
            # {walk: (grid, curves)} / {walk: {metric: values}} cells
            by_walk = {w: cbc[col]
                       for w, cbc in curves.get(conn, {}).items()
                       if col in cbc}
            vals_by_walk = {w: arr[col][metric]
                            for w, arr in box.get(conn, {}).items()
                            if col in arr}
            # --- overlay cell (left of the pair) ---
            ax = fig.add_subplot(outer[row, plot_col])
            if anchor is None:
                anchor = ax
            ax.sharex(anchor)
            cycle_axes.append(ax)
            draw_overlay(ax, by_walk, label, unit,
                         f"{conn}: {label}" if row == 0 else None)
            # --- box-plot cell (directly right of its overlay) ---
            bx = fig.add_subplot(outer[row, plot_col + 1])
            box_axes.append(bx)
            draw_boxplot(bx, vals_by_walk, metric, unit,
                         f"{conn}: {METRIC_LABEL[metric]}"
                         if row == 0 else None)
            if row == len(SIGNALS) - 1:
                bottom_axes.extend((ax, bx))

    # One common cycle-% x axis: tick labels only on the bottom row
    # (same for the box-plot walk-type labels).
    anchor.set_xlim(0.0, 100.0)
    for ax in cycle_axes:
        ax.tick_params(labelbottom=ax in bottom_axes)
    for bx in box_axes:
        bx.tick_params(labelbottom=bx in bottom_axes)
    fig.supxlabel(CYCLE_XLABEL)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE,
                    help="Root folder to search for the trial CSVs")
    ap.add_argument("--out-dir",
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="Folder for the figures and the plot-data CSVs")
    ap.add_argument("--source", choices=("PC", "SD"), default="PC",
                    help="Recording source (default: PC)")
    ap.add_argument("--gain", type=int, default=1,
                    help="Recording gain (default: 1)")
    ap.add_argument("--walk-type", choices=WALK_TYPES, default=None,
                    help="Only plot this walk type (default: all three)")
    ap.add_argument("--box-metric", choices=BOX_METRICS, default="rms",
                    help="Metric for the box-plot columns 2/4 (default: "
                         "rms = per-cycle RMS; p2p = peak-to-peak)")
    ap.add_argument("--no-plot", action="store_true",
                    help="Do not open an interactive plot window")
    args = ap.parse_args(argv)

    if plt is None:
        raise SystemExit("matplotlib is required for the plotter")

    walk_types = (args.walk_type,) if args.walk_type else WALK_TYPES
    plot_rows = []
    box_rows = []
    # curves[conn][walk_type][column] = (grid, curves_arr)
    curves = {c: {} for c in CONNECTIONS}
    # trial_ids[conn][walk_type] = trial id the curves came from
    trial_ids = {c: {} for c in CONNECTIONS}
    # box[conn][walk_type][column] = {"rms": arr, "p2p": arr} per cycle
    box = {c: {} for c in CONNECTIONS}
    columns = [c for c, _, _ in SIGNALS]

    for walk_type in walk_types:
        group = find_max_trials(args.base, walk_type, args.source,
                                args.gain)
        if not group:
            print(f"WARNING: no matching gain-{args.gain} {args.source} "
                  f"{SUBJECT} {walk_type}_Walk trials found, skipping")
            continue
        print(f"\n{walk_type} walk: {len(group)} trials: "
              + ", ".join(t for t, _, _ in group.values()))

        for connection, (trial_id, signals_path,
                         emg_path) in group.items():
            loaded = load_trial(signals_path, emg_path)
            if loaded is None:
                print(f"{trial_id}: no heel strikes, skipping")
                continue
            meta, signals, emg = loaded
            strikes = parse_heel_strikes(meta)
            cycles = list(zip(strikes[:-1], strikes[1:]))
            curves_by_col = compute_curves(meta, signals, emg, cycles,
                                           columns)
            curves[connection][walk_type] = curves_by_col
            trial_ids[connection][walk_type] = trial_id
            print(f"{trial_id}: {len(cycles)} cycles")

    for conn in CONNECTIONS:
        missing = [w for w in walk_types if w not in curves[conn]]
        if missing:
            print(f"WARNING: {conn}: missing {', '.join(missing)} data")

    # Equalize the cycle counts: every trace (overlay mean/CI and box
    # plots) uses the SAME number of gait cycles - the lowest count
    # found across all walk types and connections - so every legend
    # reports the same n.
    counts = {}
    for conn in CONNECTIONS:
        for walk, cbc in curves[conn].items():
            for grid, arr in cbc.values():
                if arr is not None:
                    counts[(conn, walk)] = arr.shape[0]
                    break
    n_common = min(counts.values()) if counts else 0
    if counts:
        print("\nCycle counts: "
              + ", ".join(f"{conn} {walk}={n}"
                          for (conn, walk), n in counts.items()))
        if max(counts.values()) > n_common:
            print(f"Equalizing every trace to n={n_common} gait cycles "
                  "(the lowest count)")

    for conn in CONNECTIONS:
        for walk, cbc in curves[conn].items():
            for col in columns:
                grid, curves_arr = cbc.get(col, (None, None))
                if curves_arr is None:
                    continue
                curves_arr = curves_arr[:n_common]
                cbc[col] = (grid, curves_arr)
                box.setdefault(conn, {}).setdefault(walk, {})[col] = {
                    "p2p": curves_arr.max(axis=1) - curves_arr.min(axis=1),
                    "rms": np.sqrt((curves_arr ** 2).mean(axis=1)),
                }

    # Mean/CI and per-cycle box-metric CSVs from the equalized data.
    for conn in CONNECTIONS:
        for walk, cbc in curves[conn].items():
            for col in columns:
                grid, curves_arr = cbc.get(col, (None, None))
                if curves_arr is None:
                    continue
                trial_id = trial_ids[conn].get(walk, "")
                n = curves_arr.shape[0]
                mean, ci, _n = mean_and_ci(curves_arr)
                for x, m, lo, hi in zip(grid, mean, mean - ci, mean + ci):
                    plot_rows.append({
                        "connection": conn,
                        "walk_type": walk,
                        "trial_id": trial_id,
                        "signal": col,
                        "cycle_pct": x,
                        "mean": m,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_cycles": n,
                    })
                for metric_name in BOX_METRICS:
                    for k, v in enumerate(
                            box[conn][walk][col][metric_name], start=1):
                        box_rows.append({
                            "connection": conn,
                            "walk_type": walk,
                            "trial_id": trial_id,
                            "signal": col,
                            "cycle": int(k),
                            "metric": metric_name,
                            "value": float(v),
                        })

    fig_path = os.path.join(args.out_dir, "gait_group_overlay.png")
    suptitle = (f"Gait-cycle overlays ({SUBJECT}): gain-{args.gain} "
                f"{args.source} recordings - slow / fast / incline walk, "
                "mean + 90% CI, boxes = "
                f"{METRIC_LABEL[args.box_metric]}")
    make_overlay_figure(curves, box, args.box_metric, suptitle, fig_path)
    print(f"\nwritten: {fig_path}")

    csv_path = os.path.join(args.out_dir, "gait_overlay_plot_data.csv")
    plot_data = pd.DataFrame(plot_rows)
    plot_data.to_csv(csv_path, index=False)
    print(f"written: {csv_path}  ({len(plot_data)} rows)")

    box_path = os.path.join(args.out_dir, "gait_overlay_box_data.csv")
    box_data = pd.DataFrame(box_rows)
    box_data.to_csv(box_path, index=False)
    print(f"written: {box_path}  ({len(box_data)} rows)")

    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
