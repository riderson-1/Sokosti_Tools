#!/usr/bin/env python3
"""Combined sustained-run analysis: sample loss, CPU load and CSV gap statistics.

Usage:
    python plot_sustained_run.py <full_logging.log> <capture.csv> [--fs 1000] [--out-dir DIR]

Produces, by default in <script_dir>/output/:

  samples_overview_<logstem>.png
      One figure with 3 panels in the same window: EMG sample rate, EMG per-interval
      drops, and per-thread CPU load (stacked, from cycle deltas).

  samples_overview_<logstem>.log
      Everything the script prints: the exact command line, the gap statistics
      (same tables as emg_isometric.py, computed from the sample-counter column
      of the CSV), the sample-loss summary, the CPU summary and the output paths.
"""

import argparse
import os
import re
import shlex
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Map raw thread names (thread struct addresses in the log) to legend labels.
THREAD_LABELS = {
    "0x200025c0": "SD thread",
    "0x200026d8": "IMU thread",
    "0x200027f0": "Log thread",
    "0x20002908": "LED thread",
    "0x20002a20": "BLE thread",
    "0x20002b38": "Acquisition thread",
}

_TS = r"\[(\d+):(\d{2}):(\d{2})\.(\d{3}),(\d{3})\]"
TS_RE = re.compile(_TS)
# A complete timestamped log line. Log lines can be spliced into the middle of
# an analyzer line, so they are stripped from each block before parsing (the
# split pieces then re-join into the original analyzer line).
LOGLINE_RE = re.compile(r"\[\d+:\d{2}:\d{2}\.\d{3},\d{3}\] <\w+> [^\r\n]*\r?\n")
THREAD_RE = re.compile(r"^\s*(.+?)\s*: STACK: .*?CPU:\s*(\d+)\s*%", re.M)
CYCLES_RE = re.compile(r"Total CPU cycles used:\s*(\d+)")
EMG_RE = re.compile(_TS + r".*EMG/s: collected=(\d+) live_q_drop=(\d+) sd_q_drop=(\d+)")
IMU_RE = re.compile(_TS + r".*IMU/s: collected=(\d+) live_q_drop=(\d+) sd_q_drop=(\d+)")


def ts_to_s(m):
    h, mi, s, ms, us = map(int, m.groups()[:5])
    return h * 3600 + mi * 60 + s + ms / 1e3 + us / 1e6


def parse_log(path):
    """Return (emg, imu, cpu_blocks).

    emg/imu: list of (t_s, collected, live_q_drop, sd_q_drop)
    cpu_blocks: list of (t_s, {thread_name: cumulative_cycles})
    """
    with open(path, errors="replace") as f:
        lines = f.read().splitlines(keepends=True)

    emg, imu = [], []
    blocks = []
    last_t = 0.0
    cur = None
    for line in lines:
        if "Thread analyze:" in line:
            cur = (last_t, [])
            blocks.append(cur)
            continue
        m = TS_RE.search(line)
        if m:
            last_t = ts_to_s(m)
        m = EMG_RE.search(line)
        if m:
            emg.append((ts_to_s(m), *map(int, m.groups()[5:8])))
        m = IMU_RE.search(line)
        if m:
            imu.append((ts_to_s(m), *map(int, m.groups()[5:8])))
        if cur is not None:
            cur[1].append(line)

    cpu = []
    for t, raw in blocks:
        text = LOGLINE_RE.sub("", "".join(raw)).replace("\r", "")
        cycles = {}
        name = None
        for line in text.split("\n"):
            m = THREAD_RE.match(line)
            if m:
                name = m.group(1)
                continue
            m = CYCLES_RE.search(line)
            if m and name:
                cycles[name] = int(m.group(1))
                name = None
        cpu.append((t, cycles))
    return emg, imu, cpu


def print_gap_statistics(csv_path, fs):
    """Gap statistics from the sample-counter column, as in emg_isometric.py."""
    idx = pd.read_csv(csv_path, usecols=[0]).iloc[:, 0].values.astype(np.int64)
    if idx.size == 0:
        sys.exit(f"CSV has no data rows: {csv_path}")
    if np.any(np.diff(idx) < 1):
        sys.exit(f"Sample counter in {csv_path} is not strictly increasing "
                 "(duplicate or out-of-order indices) - cannot compute gap statistics.")

    full_start, full_end = int(idx[0]), int(idx[-1])
    n_full = full_end - full_start + 1

    received_mask = np.zeros(n_full, dtype=bool)
    received_mask[idx - full_start] = True
    n_missing = int((~received_mask).sum())

    d_idx = np.diff(idx)
    gap_positions = np.where(d_idx != 1)[0]
    gap_lengths = [int(d_idx[g] - 1) for g in gap_positions]

    print(f"File: {csv_path}")
    print(f"Samples: {len(idx)} received, {n_missing} missing ({100.0 * n_missing / n_full:.2f}%)")
    print(f"Gaps: {len(gap_lengths)} total")
    if gap_lengths:
        gl = np.array(gap_lengths, dtype=float)
        print("\n### Gap statistics\n")
        print("| Metric | Value |")
        print("|--------|-------|")
        print(f"| Number of gaps | {len(gap_lengths)} |")
        print(f"| Samples missing | {n_missing} ({100.0 * n_missing / n_full:.2f}%) |")
        print(f"| Gap length mean (samples) | {gl.mean():.1f} +/- {gl.std():.1f} (SD) |")
        print(f"| Gap length median (samples) | {np.median(gl):.0f} |")
        print(f"| Gap length min (samples) | {gl.min():.0f} |")
        print(f"| Gap length max (samples) | {gl.max():.0f} |")
        print(f"| Gap length mean (ms @{fs} Hz) | {gl.mean() / fs * 1000:.1f} +/- {gl.std() / fs * 1000:.1f} (SD) |")
        print(f"| Gap length max (ms @{fs} Hz) | {gl.max() / fs * 1000:.0f} |")
        print("\n### Gap details\n")
        print("| Start index | Length (samples) | Length (ms) |")
        print("|-------------|------------------|-------------|")
        for g, length in zip(gap_positions, gap_lengths):
            start_idx = int(idx[g])
            print(f"| {start_idx} | {length} | {length / fs * 1000:.0f} |")
    else:
        print("No gaps detected.")


def rate(a):
    """Per-interval rates. Report lines jitter 1.0-1.23 s apart, so divide the
    per-interval counts by the real dt. A dt > 1.6 s means a report line was
    garbled/missing -> its count is unknown, not a loss."""
    dt = np.diff(a[:, 0])
    r = a[1:, 1] / dt
    r[dt > 1.6] = np.nan
    return a[1:, 0] - a[0, 0], dt, r, a[1:, 2], a[1:, 3]


def plot_overview(emg, imu, cpu, png_path):
    emg = np.array(emg)
    imu = np.array(imu)
    if len(emg) < 2:
        sys.exit("Not enough EMG/s report lines found in the log.")
    if not cpu or not any(d for _, d in cpu):
        sys.exit("No thread analyzer data found in the log.")

    te, dte, re_, lqe, sde = rate(emg)
    ti, dti, ri, lqi, sdi = rate(imu) if len(imu) >= 2 else (None,) * 5
    nom = np.nanmedian(re_)
    gap_e = dte > 1.6
    print("EMG intervals with a missing report line:", int(gap_e.sum()))

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(te, re_, lw=.8, label="EMG")
    ax[0].axhline(nom, ls="--", c="gray", lw=.8, label=f"median {nom:.0f} Hz")
    ax[0].set_ylabel("EMG samples/s\n(normalised by real dt)")
    ax[0].legend(loc="lower right")
    ax[1].bar(te, lqe, width=1, label="live queue drops")
    ax[1].bar(te, sde, width=1, bottom=lqe, label="SD queue drops")
    ax[1].set_ylabel("EMG samples dropped\nper interval")
    ax[1].legend()

    # CPU load per thread, stacked to 100 %. The analyzer's own "CPU: x %" is a
    # running average since boot, so use deltas of the cumulative cycle counters.
    # Drop incomplete (garbled) blocks.
    full = max(len(d) for _, d in cpu)
    ref = next(d for _, d in cpu if len(d) == full)  # reference thread set
    blk = [(tt - emg[0, 0], d) for tt, d in cpu if set(d) == set(ref)]
    names = list(ref)
    tc = np.array([b[0] for b in blk])
    M = np.array([[d[n] for n in names] for _, d in blk], float)
    M = np.diff(M, axis=0)
    tc = tc[1:]
    row_sum = M.sum(1, keepdims=True)
    M = 100 * M / np.where(row_sum == 0, np.nan, row_sum)
    act = [i for i, n in enumerate(names) if n != "idle" and np.nanmean(M[:, i]) >= 0.3]
    has_idle = "idle" in names
    idle = M[:, names.index("idle")] if has_idle else 0.0
    # tiny threads + ISR + rounding
    other = np.clip(100 - np.nan_to_num(M[:, act].sum(1)) - idle, 0, None)
    stack = [np.nan_to_num(M[:, act].T), other]
    if has_idle:
        stack.append(idle)
    S = np.vstack(stack)
    S *= 100 / np.maximum(S.sum(0), 100)
    lab = [THREAD_LABELS.get(names[i], names[i]) for i in act] + ["other (small threads)"]
    col = list(plt.cm.tab10.colors[:len(act)]) + ["#bbbbbb"]
    if has_idle:
        lab.append("idle")
        col.append("#eeeeee")
    ax[2].stackplot(tc, S, labels=lab, colors=col)
    ax[2].set_ylim(0, 100)
    ax[2].set_ylabel("share of thread CPU time [%]\n(cycle deltas, ISR not incl.)")
    ax[2].set_xlabel("time since boot [s]")
    ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=8)
    print("CPU blocks used/dropped:", len(blk), len(cpu) - len(blk))

    for a in ax:
        a.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=130)
    plt.close(fig)

    print("EMG drops live/sd:", lqe.sum(), sde.sum(),
          "| IMU:", lqi.sum() if lqi is not None else "-", sdi.sum() if sdi is not None else "-")
    print("intervals with any EMG drop:", int(((lqe + sde) > 0).sum()), "of", len(te))
    print("Threads plotted:", ", ".join(lab))


class _Tee:
    """Write to the console and the log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    ap = argparse.ArgumentParser(
        description="Sustained-run overview: sample loss + CPU load figure and "
                    "CSV gap statistics, saved as PNG + log file.")
    ap.add_argument("log", help="Device full-logging output (EMG/s, IMU/s and Thread analyze lines)")
    ap.add_argument("csv", help="Capture CSV; its first column (sample counter) is used for gap statistics")
    ap.add_argument("--fs", type=int, default=1000,
                    help="EMG sample rate in Hz, used for gap lengths in ms (default 1000)")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
                    help="Output directory for the PNG and the log file (default: output/ next to this script)")
    args = ap.parse_args()

    for path, what in ((args.log, "log"), (args.csv, "CSV")):
        if not os.path.isfile(path):
            sys.exit(f"{what} file not found: {path}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.basename(args.log).rsplit(".", 1)[0]
    png_path = os.path.join(args.out_dir, f"samples_overview_{stem}.png")
    log_path = os.path.join(args.out_dir, f"samples_overview_{stem}.log")

    log_f = open(log_path, "w", buffering=1)
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, log_f)
    try:
        # record the exact command so the run can be repeated immediately:
        print("# Command: python " + " ".join(shlex.quote(a) for a in sys.argv))
        print(f"Log: {args.log}")
        print(f"CSV: {args.csv}")
        print()
        print("## Gap statistics")
        print_gap_statistics(args.csv, args.fs)
        print()
        print("## Sample loss and CPU load")
        emg, imu, cpu = parse_log(args.log)
        plot_overview(emg, imu, cpu, png_path)
        print()
        print("## Outputs")
        print("Saved figure:", png_path)
        print("Saved log:", log_path)
    finally:
        sys.stdout = orig_stdout
        log_f.close()


if __name__ == "__main__":
    main()
