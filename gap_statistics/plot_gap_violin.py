#!/usr/bin/env python3
# Violin plot of samples lost per trial (missing %) from the combined
# gap-statistics CSV: one violin each for SD card, USB PC and BLE PC.
# Saves to <out-dir>/gap_violin.png (default: <script dir>/outputs).

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUP_ORDER = ["SD card", "USB PC", "BLE PC"]


def read_summary_section(path):
    """Return the summary section of the combined CSV as dicts."""
    with open(path) as f:
        lines = f.read().splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("#")]
    if len(starts) < 2:
        sys.exit(f"summary/details sections not found in {path}")
    rows = list(csv.reader(lines[starts[0] + 1:starts[1]]))
    header, data = rows[0], rows[1:]
    return [dict(zip(header, r)) for r in data if r]


def main():
    ap = argparse.ArgumentParser(
        description="Violin plot of missing-sample %% per trial, grouped by "
                    "recording type (SD card / USB PC / BLE PC).")
    ap.add_argument("--csv", required=True,
                    help="Combined gap-statistics CSV (gap_statistics_combined.csv).")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "outputs"),
                    help="Output folder for the PNG (default: <script dir>/outputs).")
    args = ap.parse_args()

    summary = read_summary_section(args.csv)

    groups = {g: [] for g in GROUP_ORDER}
    # per-trial (samples_received, fs)
    samples = {g: [] for g in GROUP_ORDER}
    for r in summary:
        pct = float(r["missing_pct"])
        received = int(r["samples_received"])
        fs = int(r["fs"]) if r["fs"] else 980
        if r["recording"] == "SD":
            group = "SD card"
        elif r["config"].startswith("USB"):
            group = "USB PC"
        elif r["config"].startswith("BLE"):
            group = "BLE PC"
        else:
            sys.exit(f"cannot classify trial: config={r['config']!r} "
                     f"recording={r['recording']!r}")
        groups[group].append(pct)
        samples[group].append((received, fs))

    def tick_label(g):
        total_samples = sum(recv for recv, _ in samples[g])
        secs = sum(recv / fs for recv, fs in samples[g])
        time_str = (f"{secs:.0f} s ({secs/60:.1f} min)" if secs >= 60
                    else f"{secs:.0f} s")
        return (f"{g}\nn={len(groups[g])}, {total_samples:,} samples\n"
                f"≈{time_str}")

    for g in GROUP_ORDER:
        if len(groups[g]) < 2:
            sys.exit(f"group {g!r} has only {len(groups[g])} point(s); "
                     "need at least 2 for a violin")

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [groups[g] for g in GROUP_ORDER]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for j, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
        if j == 0:
            pc.set_label("Distribution (violin)")
    parts["cmeans"].set_color("crimson")
    parts["cmeans"].set_label("Mean")
    parts["cmaxes"].set_color("0.4")
    parts["cmaxes"].set_linestyle("--")
    parts["cmins"].set_color("0.4")
    parts["cmins"].set_linestyle("--")
    parts["cmaxes"].set_label("Max / min")

    ax.set_xticks(range(1, len(GROUP_ORDER) + 1))
    ax.set_xticklabels([tick_label(g) for g in GROUP_ORDER])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylabel("Samples lost per trial (% missing)")
    ax.set_title("Sample loss per trial by recording type")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)

    # annotate the mean value next to each mean marker
    for i, g in enumerate(GROUP_ORDER, start=1):
        vals = groups[g]
        mean = sum(vals) / len(vals)
        ax.annotate(f"{mean:.2f}%", xy=(i, mean), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    color="crimson", fontweight="bold")

    os.makedirs(args.out_dir, exist_ok=True)
    png_path = os.path.join(args.out_dir, "gap_violin.png")
    pdf_path = os.path.join(args.out_dir, "gap_violin.pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    for g in GROUP_ORDER:
        vals = groups[g]
        total_samples = sum(recv for recv, _ in samples[g])
        secs = sum(recv / fs for recv, fs in samples[g])
        print(f"{g}: n={len(vals)} samples={total_samples:,} "
              f"duration={secs:.0f} s ({secs/60:.1f} min) "
              f"mean={sum(vals)/len(vals):.2f}% "
              f"min={min(vals):.2f}% max={max(vals):.2f}%")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()