#!/usr/bin/env python3
"""
Data processing stage of the crosstalk analysis pipeline.
Loads all PC.csv files, extracts amplitudes, computes crosstalk matrices,
and saves results to an intermediary NPZ file for plotting.

Usage:
    python3 analyze.py
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

BASELINE_PATH = "/home/karl/Documents/Master_Thesis/Testing/benchtop_testing/breakout_pcb_baseline"
CROSSTALK_PATH = "/home/karl/Documents/Master_Thesis/Testing/benchtop_testing/crosstalk_test"
OUTPUT_DIR = "/home/karl/repos/Sokosti_tools/benchtop_testing_tools/crosstalk_test/outputs"

FS = 980  # effective ADS1299 sample rate (2.000 MHz clock instead of 2.048 MHz -> ~980 SPS, measured)
GAIN = 1
VREF = 4.5
N_CHANNELS = 16
LSB_V = 2.0 * VREF / (GAIN * 2**24)
LSB_uV = LSB_V * 1e6

SEGMENT_DURATION = 3.0
SINE_F0 = [10, 100]
AMPLITUDES_MVPP = [2, 10]

NOISE_CORR_THRESHOLD_DB = 6.0
SIGNIFICANT_LOSS_FRAC = 0.01

CHIP_OF = {ch: ("ads1299-1" if 1 <= ch <= 8 else "ads1299-2") for ch in range(1, 17)}

# Channels whose driven-channel PSD is saved for the PSD overlay figure
PSD_CHANNELS = [1, 4, 8, 9, 13, 16]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
})

FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
INTERIM_PATH = os.path.join(OUTPUT_DIR, "interim_data.npz")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_pc_csv(filepath):
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
        "n_gaps": n_gaps, "max_gap_samples": max_gap, "total_lost_samples": total_lost,
        "flagged": flagged, "start_idx": start_idx, "end_idx": end_idx,
        "n_samples_full": n, "n_samples_seg": len(df_seg),
    }
    return df_seg, gap_info, df_full


def parse_filename(filepath):
    fname = os.path.basename(filepath)
    m = re.match(r"CH(\d+)_1G_Sine_(\d+)mVpp_(\d+)Hz_PC\.csv", fname)
    if not m:
        raise ValueError(f"Cannot parse filename: {fname}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ============================================================================
# SIGNAL CONDITIONING
# ============================================================================

def counts_to_uv(counts):
    return counts * LSB_uV


def remove_dc(signal_uv):
    return signal_uv - np.mean(signal_uv)


def split_into_segments(signal_uv, fs, seg_dur):
    n_seg_samples = int(fs * seg_dur)
    n_full = (len(signal_uv) // n_seg_samples) * n_seg_samples
    if n_full == 0:
        return []
    return [(signal_uv[i:i + n_seg_samples], i) for i in range(0, n_full, n_seg_samples)]


# ============================================================================
# AMPLITUDE EXTRACTION
# ============================================================================

def amplitude_ls_sine_fit(t, y, fs, f0, f_bound_hz=2.0):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.std(y) < 1e-12:
        return {"amp_peak_uV": np.nan, "freq_hz": np.nan, "phase_rad": np.nan, "offset_uV": np.nan}

    def residuals(p):
        A, f, phi, C = p
        if A <= 0 or not (f0 - f_bound_hz <= f <= f0 + f_bound_hz):
            return np.full_like(y, 1e12)
        return y - (A * np.sin(2 * np.pi * f * t + phi) + C)

    # Initialization: at fixed frequency the model is linear in (a, b, C), so a coarse
    # frequency grid with closed-form 3x3 least-squares solves finds the right basin
    # (amplitude, phase AND frequency) without any nonlinear iteration. The 4-parameter
    # fit then polishes from that start in a few iterations. This avoids the phase
    # local-minimum trap that a naive phi0=0 start hits.
    n_grid = 81
    f_grid = np.linspace(f0 - f_bound_hz, f0 + f_bound_hz, n_grid)
    best_cost, best_p = np.inf, None
    for fg in f_grid:
        X = np.column_stack([np.sin(2 * np.pi * fg * t), np.cos(2 * np.pi * fg * t), np.ones_like(t)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ coef
        cost = float(r @ r)
        if cost < best_cost:
            best_cost, best_p = cost, (coef[0], coef[1], fg)
    a, b, f_init = best_p
    A0 = np.hypot(a, b)
    phi0 = np.arctan2(b, a)  # y ~ A*sin(2*pi*f*t + phi) with a = A*cos(phi), b = A*sin(phi)
    p0 = [A0, f_init, np.clip(phi0, -np.pi, np.pi), np.mean(y)]
    bounds = ([0, f0 - f_bound_hz, -np.pi, -np.inf],
              [np.inf, f0 + f_bound_hz, np.pi, np.inf])
    try:
        res = least_squares(residuals, p0, bounds=bounds, method="trf", max_nfev=500,
                            xtol=1e-6, ftol=1e-6, gtol=1e-6)
        return {"amp_peak_uV": float(res.x[0]), "freq_hz": float(res.x[1]),
                "phase_rad": float(res.x[2]), "offset_uV": float(res.x[3])}
    except Exception:
        return {"amp_peak_uV": np.nan, "freq_hz": np.nan, "phase_rad": np.nan, "offset_uV": np.nan}


def amplitude_fft_hann(y, fs, f0):
    y = np.asarray(y, dtype=float)
    N = len(y)
    if N == 0:
        return np.nan
    w = np.hanning(N)
    spectrum = np.fft.rfft(y * w)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    k = np.argmin(np.abs(freqs - f0))
    sum_w = np.sum(w)
    if sum_w == 0:
        return np.nan
    return float(2.0 * np.abs(spectrum[k]) / sum_w)


def amplitude_rms_sqrt2(y):
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(2.0) * np.sqrt(np.mean(y**2)))


# ============================================================================
# NOISE REFERENCE FROM BASELINE
# ============================================================================

def compute_noise_reference(baseline_seg_df, fs, freqs):
    n_seg_samples = int(fs * SEGMENT_DURATION)
    noise_ref = {}
    for ch in range(1, N_CHANNELS + 1):
        ch_uV = remove_dc(counts_to_uv(baseline_seg_df[f"ch{ch}"].values.astype(float)))
        segs = split_into_segments(ch_uV, fs, SEGMENT_DURATION)
        if not segs:
            continue

        fft_amps = {}
        for f0 in freqs:
            amps = [amplitude_fft_hann(seg, fs, f0) for seg, _ in segs]
            amps = [a for a in amps if np.isfinite(a)]
            fft_amps[f0] = float(np.median(amps)) if amps else np.nan

        nperseg = min(2048, len(ch_uV))
        freqs_welch, psd_welch = sp_signal.welch(ch_uV, fs=fs, window="hann", nperseg=nperseg,
                                                  noverlap=nperseg // 2, scaling="density")

        noise_ref[ch] = {
            "fft_amps_uV": fft_amps,
            "welch_freqs_hz": freqs_welch,
            "welch_psd_uv2": psd_welch,
            "mean_uV": float(np.mean(ch_uV)),
            "rms_uV": float(np.sqrt(np.mean(ch_uV**2))),
            "vpp_uV": float(np.ptp(ch_uV)),
        }
    return noise_ref


# ============================================================================
# CROSSTALK METRICS
# ============================================================================

def compute_crosstalk_row(driven_amp_uV, victim_amps_uV_array, noise_amp_uV, threshold_db=6.0):
    """Crosstalk from one driven channel to all 16 victims (1-D, length 16)."""
    if not np.isfinite(driven_amp_uV) or driven_amp_uV <= 0:
        return {
            "xt_db": np.full(16, np.nan),
            "xt_corrected": np.full(16, np.nan),
            "censored": np.ones(16, dtype=bool),
            "detection_limit_db": np.nan,
        }

    xt_db = 20.0 * np.log10(victim_amps_uV_array / driven_amp_uV)
    xt_db = np.where(np.isfinite(xt_db), xt_db, np.nan)
    detection_limit = 20.0 * np.log10(noise_amp_uV / driven_amp_uV)

    threshold_linear = 10 ** (threshold_db / 20.0)
    can_correct = victim_amps_uV_array >= noise_amp_uV * threshold_linear
    A_ct = np.where(can_correct, np.sqrt(np.maximum(victim_amps_uV_array**2 - noise_amp_uV**2, 0)), np.nan)
    xt_corrected = 20.0 * np.log10(A_ct / driven_amp_uV)
    xt_corrected = np.where(np.isfinite(xt_corrected), xt_corrected, np.nan)

    censored = ~can_correct

    return {
        "xt_db": xt_db, "xt_corrected": xt_corrected, "censored": censored,
        "detection_limit_db": detection_limit,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Analysis Stage: Data Processing")
    print("=" * 70)

    # --- Load baseline ---
    print("\n[1/4] Loading baseline...")
    baseline_file = os.path.join(BASELINE_PATH, "sokosti_ble_capture_20260915_151745_rec01.csv")
    baseline_seg, baseline_gaps, _ = load_pc_csv(baseline_file)
    baseline_duration = baseline_gaps["n_samples_seg"] / FS
    print(f"  Baseline: {baseline_gaps['n_samples_full']:,} samples, "
          f"{baseline_gaps['n_samples_seg']:,} gap-free, {baseline_duration:.2f} s")

    # --- Compute noise reference ---
    print("\n[2/4] Computing noise reference...")
    noise_ref = compute_noise_reference(baseline_seg, FS, SINE_F0)
    print(f"  Noise reference computed for {len(noise_ref)} channels")

    # --- Process crosstalk files ---
    print("\n[3/4] Processing crosstalk files...")
    pc_files = sorted(glob.glob(os.path.join(CROSSTALK_PATH, "*", "*_PC.csv")))
    print(f"  Found {len(pc_files)} PC.csv files")

    condition_data = {}
    driven_psds = {}
    log_entries = []
    files_with_significant_loss = []
    failed_files = []

    for filepath in pc_files:
        try:
            ch, amp, f0 = parse_filename(filepath)
            df_seg, gaps, _ = load_pc_csv(filepath)
            duration = len(df_seg) / FS

            ch_data_uV = {}
            for c in range(1, 17):
                ch_data_uV[c] = remove_dc(counts_to_uv(df_seg[f"ch{c}"].values.astype(float)))

            n_seg_samples = int(FS * SEGMENT_DURATION)
            n_full = (len(ch_data_uV[1]) // n_seg_samples) * n_seg_samples
            n_segments = n_full // n_seg_samples
            max_segs = 3
            if n_segments > max_segs:
                n_segments = max_segs

            seg_ls = {c: [] for c in range(1, 17)}
            for seg_idx in range(n_segments):
                start = seg_idx * n_seg_samples
                end = start + n_seg_samples
                t = np.arange(n_seg_samples) / FS
                for c in range(1, 17):
                    seg = ch_data_uV[c][start:end]
                    ls = amplitude_ls_sine_fit(t, seg, FS, f0)
                    seg_ls[c].append(ls["amp_peak_uV"])

            # Welch PSDs of all 16 channels (for the PSD overlay figure)
            # Key includes the driven channel so PSDs from each driven
            # channel's own recording are kept separately.
            if ch in PSD_CHANNELS and (f0, amp, ch) not in driven_psds:
                nperseg = min(2048, len(ch_data_uV[1]))
                for c in range(1, 17):
                    f_psd, psd = sp_signal.welch(ch_data_uV[c], fs=FS, window="hann",
                                                 nperseg=nperseg, noverlap=nperseg // 2,
                                                 scaling="density")
                    driven_psds[(f0, amp, ch, c)] = (f_psd, psd)

            amp_per_channel = {}
            for c in range(1, 17):
                ls_vals = [v for v in seg_ls[c] if np.isfinite(v)]
                amp_per_channel[c] = float(np.median(ls_vals)) if ls_vals else np.nan

            key = (f0, amp)
            if key not in condition_data:
                condition_data[key] = {
                    "driven_channel": ch,
                    "per_channel_amps": {},
                    "n_segments": n_segments,
                    "duration_s": duration,
                }
            condition_data[key]["per_channel_amps"][ch] = amp_per_channel

            log_entries.append({
                "file": os.path.basename(filepath), "channel": ch, "freq": f0, "amp": amp,
                "n_samples": len(df_seg), "duration_s": duration,
                "n_gaps": gaps["n_gaps"], "max_gap": gaps["max_gap_samples"],
                "lost": gaps["total_lost_samples"], "flagged": gaps["flagged"],
            })
            if gaps["flagged"]:
                files_with_significant_loss.append(os.path.basename(filepath))

        except Exception as e:
            failed_files.append((filepath, str(e)))
            print(f"  FAILED: {os.path.basename(filepath)}: {e}")

        done = len(log_entries) + len(failed_files)
        if done % 10 == 0 or done == len(pc_files):
            print(f"  Processed {done}/{len(pc_files)} files...")

    # --- Build crosstalk matrices ---
    # Each (f0, amp) condition has 16 files (one per driven channel); each file
    # contributes one row of the 16x16 matrix (XT from that driven ch to all victims).
    print("\n[4/4] Building crosstalk matrices and saving interim data...")
    crosstalk_results = {}
    for key, data in condition_data.items():
        f0, amp = key
        xt_db = np.full((16, 16), np.nan)
        xt_corrected = np.full((16, 16), np.nan)
        censored = np.ones((16, 16), dtype=bool)
        driven_amps = {}
        detection_limits = {}
        for driven_ch, file_amps in data["per_channel_amps"].items():
            A_driven = file_amps[driven_ch]
            victim_amps = np.array([file_amps[c] for c in range(1, 17)])
            A_noise = noise_ref[driven_ch]["fft_amps_uV"].get(f0, np.nan)
            row = compute_crosstalk_row(A_driven, victim_amps, A_noise)
            xt_db[driven_ch - 1] = row["xt_db"]
            xt_corrected[driven_ch - 1] = row["xt_corrected"]
            censored[driven_ch - 1] = row["censored"]
            censored[driven_ch - 1, driven_ch - 1] = True
            xt_db[driven_ch - 1, driven_ch - 1] = np.nan
            xt_corrected[driven_ch - 1, driven_ch - 1] = np.nan
            driven_amps[driven_ch] = float(A_driven) if np.isfinite(A_driven) else np.nan
            detection_limits[driven_ch] = float(row["detection_limit_db"]) if np.isfinite(row["detection_limit_db"]) else np.nan

        crosstalk_results[key] = {
            "frequency": f0, "amplitude": amp,
            "driven_amps_uV": driven_amps, "detection_limits_db": detection_limits,
            "xt_db": xt_db, "xt_corrected": xt_corrected, "censored": censored,
            "n_segments": data["n_segments"], "duration_s": data["duration_s"],
        }

    # --- Save interim data ---
    print("\nSaving interim data...")

    # Save crosstalk matrices as structured arrays for easy loading
    matrices_data = []
    for key, res in crosstalk_results.items():
        matrices_data.append((
            res["frequency"], res["amplitude"],
            res["xt_db"].astype(np.float32), res["xt_corrected"].astype(np.float32),
            res["censored"].astype(bool),
            res["driven_amps_uV"], res["detection_limits_db"],
        ))

    # Save noise reference
    noise_data = []
    for ch in range(1, N_CHANNELS + 1):
        if ch not in noise_ref:
            continue
        nr = noise_ref[ch]
        noise_data.append((
            ch,
            nr["fft_amps_uV"].get(10, np.nan), nr["fft_amps_uV"].get(100, np.nan),
            nr["welch_freqs_hz"], nr["welch_psd_uv2"],
            nr["mean_uV"], nr["rms_uV"], nr["vpp_uV"],
        ))

    # Save log entries
    log_df = pd.DataFrame(log_entries)

    # Save baseline info
    baseline_info = {
        "n_samples_full": baseline_gaps["n_samples_full"],
        "n_samples_seg": baseline_gaps["n_samples_seg"],
        "n_gaps": baseline_gaps["n_gaps"],
        "max_gap_samples": baseline_gaps["max_gap_samples"],
        "total_lost_samples": baseline_gaps["total_lost_samples"],
        "duration_s": baseline_duration,
    }

    # Save as NPZ for fast loading
    np.savez(INTERIM_PATH,
             matrices_n_f0=np.array([m[0] for m in matrices_data]),
             matrices_n_amp=np.array([m[1] for m in matrices_data]),
             matrices_xt_db=np.array([m[2] for m in matrices_data]),
             matrices_xt_corr=np.array([m[3] for m in matrices_data]),
             matrices_censored=np.array([m[4] for m in matrices_data]),
             matrices_driven_amps=np.array([m[5] for m in matrices_data], dtype=object),
             matrices_detection_limits=np.array([m[6] for m in matrices_data], dtype=object),
             noise_ch=np.array([m[0] for m in noise_data]),
             noise_fft10=np.array([m[1] for m in noise_data]),
             noise_fft100=np.array([m[2] for m in noise_data]),
             noise_welch_freqs=[m[3] for m in noise_data],
             noise_welch_psd_uv2=[m[4] for m in noise_data],
             noise_mean_uV=np.array([m[5] for m in noise_data]),
             noise_rms_uV=np.array([m[6] for m in noise_data]),
             noise_vpp_uV=np.array([m[7] for m in noise_data]),
             psd_keys=np.array([f"{f0}_{amp}_{ch}_{c}" for (f0, amp, ch, c) in driven_psds.keys()], dtype=object),
             psd_freqs=np.array([v[0] for v in driven_psds.values()], dtype=object),
             psd_density_nv=np.array([v[1] for v in driven_psds.values()], dtype=object),
             baseline_info=baseline_info,
             )

    # Save log as CSV
    log_df.to_csv(os.path.join(TABLE_DIR, "file_log.csv"), index=False)

    # Save crosstalk matrices as CSV (long format)
    crosstalk_rows = []
    for key, res in crosstalk_results.items():
        f0, amp = key
        for i in range(16):
            for j in range(16):
                crosstalk_rows.append({
                    "freq_hz": f0, "amp_mvpp": amp,
                    "driven_ch": i + 1,
                    "victim_ch": j + 1,
                    "xt_db": res["xt_db"][i, j],
                    "xt_corrected": res["xt_corrected"][i, j],
                    "censored": bool(res["censored"][i, j]),
                })
    crosstalk_df = pd.DataFrame(crosstalk_rows)
    crosstalk_df.to_csv(os.path.join(TABLE_DIR, "crosstalk_matrices_long.csv"), index=False)

    # Save noise reference as CSV
    noise_rows = []
    for ch in range(1, N_CHANNELS + 1):
        if ch not in noise_ref:
            continue
        nr = noise_ref[ch]
        noise_rows.append({
            "channel": ch,
            "mean_uV": nr["mean_uV"],
            "rms_uV": nr["rms_uV"],
            "vpp_uV": nr["vpp_uV"],
            "fft_amp_10Hz_uV": nr["fft_amps_uV"].get(10, np.nan),
            "fft_amp_100Hz_uV": nr["fft_amps_uV"].get(100, np.nan),
        })
    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(os.path.join(TABLE_DIR, "noise_reference.csv"), index=False)

    print(f"\nSaved interim data to: {INTERIM_PATH}")
    print(f"Saved file log: {os.path.join(TABLE_DIR, 'file_log.csv')}")
    print(f"Saved crosstalk matrices: {os.path.join(TABLE_DIR, 'crosstalk_matrices_long.csv')}")
    print(f"Saved noise reference: {os.path.join(TABLE_DIR, 'noise_reference.csv')}")

    print(f"\nFiles processed: {len(log_entries)}")
    print(f"Failed files: {len(failed_files)}")
    if failed_files:
        for f, e in failed_files:
            print(f"  - {os.path.basename(f)}: {e}")
    print(f"Files with significant packet loss: {len(files_with_significant_loss)}")

    print("\nAnalysis stage complete. Run plot.py to generate figures.")


if __name__ == "__main__":
    main()
