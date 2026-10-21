#!/usr/bin/env python3
"""
Data processing stage of the standalone-device baseline analysis.

The standalone device baseline is recorded with nothing connected to the
front end (no breakout board, no electrodes, mux = 001). This script loads
the BLE-streamed PC recording, handles BLE packet loss, converts ADC counts
to microvolts, and computes the per-channel noise metrics that the reference
crosstalk baseline pipeline produces.

Usage:
    python3 analyze.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# CONFIGURATION CONSTANTS (identical to the crosstalk reference pipeline)
# ============================================================================

DEVICE_BASELINE_PATH = "/home/karl/Documents/Master_Thesis/Testing/benchtop_testing/device_baseline"
OUTPUT_DIR = "/home/karl/repos/Sokosti_tools/benchtop_testing_tools/standalone_baseline_test/outputs"

FS = 1000            # sampling rate, Hz
GAIN = 1             # PGA gain (1G in the folder / file name)
VREF = 4.5           # reference voltage, V
N_CHANNELS = 16
LSB_V = 2.0 * VREF / (GAIN * 2 ** 24)   # ADC LSB in volts
LSB_uV = LSB_V * 1e6

SEGMENT_DURATION = 3.0                  # analysis segment length, s
SIGNIFICANT_LOSS_FRAC = 0.01            # packet-loss flag threshold

FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
INTERIM_PATH = os.path.join(OUTPUT_DIR, "interim_data.npz")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_pc_csv(filepath):
    """Load a PC recording, detect BLE gaps via the sample counter, and
    return the longest gap-free segment (same approach as the reference)."""
    df_full = pd.read_csv(filepath)
    samples = df_full["sample"].values.astype(np.int64)
    n = len(df_full)
    diffs = np.diff(samples)
    gaps = diffs - 1
    gap_indices = np.where(gaps > 0)[0]
    n_gaps = len(gap_indices)
    total_lost = int(gaps.sum())
    max_gap = int(gaps.max()) if n_gaps > 0 else 0

    if n_gaps == 0:
        start_idx, end_idx = 0, n
    else:
        starts = np.concatenate([[0], gap_indices + 1])
        ends = np.concatenate([gap_indices + 1, [n]])
        lengths = ends - starts
        best = np.argmax(lengths)
        start_idx, end_idx = int(starts[best]), int(ends[best])

    df_seg = df_full.iloc[start_idx:end_idx].reset_index(drop=True)
    flagged = total_lost > SIGNIFICANT_LOSS_FRAC * (samples[-1] - samples[0] + 1)

    gap_info = {
        "n_gaps": n_gaps, "max_gap_samples": max_gap,
        "total_lost_samples": total_lost, "flagged": flagged,
        "start_idx": start_idx, "end_idx": end_idx,
        "n_samples_full": n, "n_samples_seg": len(df_seg),
    }
    return df_seg, gap_info, df_full


# ============================================================================
# SIGNAL CONDITIONING
# ============================================================================

def counts_to_uv(counts):
    return counts * LSB_uV


def remove_dc(signal_uv):
    return signal_uv - np.mean(signal_uv)


def split_into_segments(signal_uv, fs, seg_dur):
    """Split into non-overlapping segments of seg_dur seconds."""
    n_seg_samples = int(fs * seg_dur)
    n_full = (len(signal_uv) // n_seg_samples) * n_seg_samples
    if n_full == 0:
        return []
    return [(signal_uv[i:i + n_seg_samples], i) for i in range(0, n_full, n_seg_samples)]


# ============================================================================
# NOISE METRICS
# ============================================================================

def compute_baseline_noise(df_seg, fs):
    """Per-channel noise metrics from the gap-free baseline segment.

    Returns a dict per channel with:
      - mean_uV, rms_uV, vpp_uV over the whole gap-free section
        (rms_uV covers the whole spectrum, 0 to Nyquist),
      - welch_freqs_hz / welch_psd_uv2: PSD in uV^2/Hz
        (Welch, scaling='density', psd already in uV^2/Hz).
    """
    noise = {}
    for ch in range(1, N_CHANNELS + 1):
        ch_uV = remove_dc(counts_to_uv(df_seg[f"ch{ch}"].values.astype(float)))

        nperseg = min(2048, len(ch_uV))
        freqs_psd, psd = sp_signal.welch(ch_uV, fs=fs, window="hann",
                                         nperseg=nperseg, noverlap=nperseg // 2,
                                         scaling="density")

        noise[ch] = {
            "mean_uV": float(np.mean(ch_uV)),
            "rms_uV": float(np.sqrt(np.mean(ch_uV ** 2))),
            "vpp_uV": float(np.ptp(ch_uV)),
            "welch_freqs_hz": freqs_psd,
            "welch_psd_uv2": psd,
        }
    return noise


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Analysis Stage: Standalone Device Baseline")
    print("=" * 70)

    csv_files = sorted(glob.glob(os.path.join(DEVICE_BASELINE_PATH, "*.csv")))
    if len(csv_files) != 1:
        raise RuntimeError(f"Expected exactly 1 CSV in {DEVICE_BASELINE_PATH}, found {len(csv_files)}")

    baseline_file = csv_files[0]
    df_seg, gap_info, _ = load_pc_csv(baseline_file)
    duration = gap_info["n_samples_seg"] / FS

    print(f"\n  File: {os.path.basename(baseline_file)}")
    print(f"  Full recording: {gap_info['n_samples_full']:,} samples")
    print(f"  Gaps: {gap_info['n_gaps']} total, "
          f"max {gap_info['max_gap_samples']} samples, "
          f"{gap_info['total_lost_samples']:,} lost "
          f"({100.0 * gap_info['total_lost_samples'] / gap_info['n_samples_full']:.3f}%)")
    print(f"  Longest gap-free section: {gap_info['n_samples_seg']:,} samples "
          f"({duration:.2f} s), rows {gap_info['start_idx']}..{gap_info['end_idx']}")
    if gap_info["flagged"]:
        print("  WARNING: significant packet loss detected (flagged).")

    n_segments = len(split_into_segments(np.zeros(int(duration * FS)), FS, SEGMENT_DURATION))
    print(f"  Full {SEGMENT_DURATION:.0f} s segments in gap-free section: {n_segments}")

    noise = compute_baseline_noise(df_seg, FS)
    print(f"  Computed noise metrics for {len(noise)} channels")

    # ---- Save interim data ----
    ch_arr = np.array(list(noise.keys()))
    mean_uV = np.array([noise[c]["mean_uV"] for c in noise])
    rms_uV = np.array([noise[c]["rms_uV"] for c in noise])
    vpp_uV = np.array([noise[c]["vpp_uV"] for c in noise])
    welch_freqs = [noise[c]["welch_freqs_hz"] for c in noise]
    welch_psd = [noise[c]["welch_psd_uv2"] for c in noise]

    np.savez(INTERIM_PATH,
             noise_ch=ch_arr,
             noise_mean_uV=mean_uV,
             noise_rms_uV=rms_uV,
             noise_vpp_uV=vpp_uV,
             noise_welch_freqs=welch_freqs,
             noise_welch_psd_uv2=welch_psd,
             gap_info=gap_info,
             duration_s=duration,
             )
    print(f"\n  Saved interim data: {INTERIM_PATH}")


if __name__ == "__main__":
    main()
