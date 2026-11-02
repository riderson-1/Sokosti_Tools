#!/usr/bin/env python3
"""
PCA-based sagittal-plane isolation for Sokosti IMU walking captures.

============================================================================
DISCLAIMER - accelerometer scaling in captures recorded before 2026-10-02
============================================================================
Captures recorded before the 2026-10-02 fix in sokosti/protocol.py stored
the BHI360 accelerometer with a WRONG conversion (raw/16 instead of the
correct raw/4096, per the Bosch BHY2-Sensor-API default scaling table).
In those files the accel_x/y/z columns are NOT g: 1 g reads as ~256.
The columns also include gravity: the firmware streams sensor ID 4, the
*corrected accelerometer*, not linear acceleration (that would be ID 31).

This script auto-detects the legacy scale (median |accel| >> 1 g) and
rescales by 1/256 so everything downstream is in g. Roll/pitch/yaw are
derived from the quaternion (Q14, /16384) and were always correct.
Override auto-detection with --accel-unit {auto,g,legacy}.

Pipeline
--------
1. Load (sample, roll, pitch, yaw, accel_x/y/z) from the capture CSV.
2. Resample the 50 Hz zero-order-held IMU streams onto a uniform grid.
3. No IMU filters: the angles are used raw (unwrapped only), and gravity
   is removed from the accelerometer by subtracting the mean gravity
   vector (DC removal) -> dynamic acceleration.
4. PCA:
   - Angular-rate PCA on [d(roll)/dt, d(pitch)/dt, d(yaw)/dt]: PC1 is the
     dominant rotation axis. For sagittal-dominant walking PC1 should align
     with the pitch axis and explain >~0.8 of the variance.
   - Dynamic-accel PCA: PC1/PC2 should span the sagittal (forward+vertical)
     plane; a small PC3 fraction means the motion is mostly planar.
5. Signal Vector Magnitude (SVM), orientation-independent motion intensity:
   - |omega| = sqrt(wx^2 + wy^2 + wz^2) from the angular rates.
   - |a_dyn| = sqrt(ax^2 + ay^2 + az^2) from the dynamic (gravity-removed)
     acceleration. No "-1 g" term is needed here: that ENMO convention
     applies to the raw accel magnitude, but gravity is already removed
     by the high-pass in step 3.
   SVM complements the PCA: it says HOW MUCH total motion, the PCA says
   WHICH plane it lives in. SVM cannot distinguish sagittal from frontal
   motion on its own.
6. Project, canonicalise the arbitrary PCA sign (angular-rate PC1:
   pitch-up = positive; dynamic-accel PC1: largest axis component
   positive), optionally plot and export the processed signals.
7. Second figure: EMG linear envelope (ch1 by default, --channel) computed
   with the SAME filter chain as plot_isometric_raw_modes.py (bandpass
   20-450 Hz, NO notch, rectify, low-pass, all zero-phase, gap-aware per
   contiguous valid segment) above the dynamic-accel PC1 and the
   angular-rate PC1 on one shared time axis, so muscle activity and
   sagittal-plane motion can be compared on the same timeline.
8. Third figure (--heel-strikes "t1,t2,..."): gait-cycle-locked overlays.
   Consecutive heel strikes bound one gait cycle (n strikes -> n-1 cycles;
   the first and last strike mark the start and finish). Each cycle is
   time-normalized to 0-100 percent (default) or kept in real seconds from
   heel strike (--time-axis time). The EMG envelope, dynamic-accel PC1 and
   angular-rate PC1 are overlaid as individual traces + mean + 90 % CI,
   like the contraction overlays in emg_isometric.py.

Usage
-----
python pca_sagittal_plane_isolation.py CAPTURE.csv

Sample rate: the captures run at 980 sps (NOT 1000 -- same as the Iso_Dor
and Sit_Stand trials), so --emg-fs defaults to 980. All time values passed
on the command line (--heel-strikes, --fit-start/--fit-end) live in that
time base; times recorded against the old fs=1000 axis must be scaled by
1000/980 to select the same samples.
python pca_sagittal_plane_isolation.py CAPTURE.csv --fit-start 10 --fit-end 60
python pca_sagittal_plane_isolation.py CAPTURE.csv --no-plot --save-csv out.csv
python pca_sagittal_plane_isolation.py CAPTURE.csv --channel 2 --save-results
python pca_sagittal_plane_isolation.py CAPTURE.csv --heel-strikes 10.5,12.3,14.1
"""

import argparse
import os
import shlex
import sys

import numpy as np
import pandas as pd
from scipy import signal

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None

IMU_COLS = ("roll", "pitch", "yaw", "accel_x", "accel_y", "accel_z")
ANGLE_NAMES = ("roll(x)", "pitch(y)", "yaw(z)")
# Legacy CSVs stored raw/16; correct is raw/4096 g -> stored/256 = g.
LEGACY_TO_G = 1.0 / 256.0


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def pca(X):
    """
    X: (N, k) observation matrix (rows = samples), e.g. a 3-axis signal.

    Returns (components, explained):
      components: (k, k), PC1 in column 0, ordered by descending eigenvalue.
      explained:  variance fraction per component (sums to 1).
    """
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    return eigvecs, eigvals / eigvals.sum()


def fix_sign_rate(rate_axis, rate_pc1):
    """Canonical polarity for angular-rate PC1: pitch-up = positive.

    The PC sign is arbitrary; anchoring it on the pitch component of
    the PC1 axis (large and stable for sagittal-dominant motion) keeps
    PC1 comparable across trials, connections and sources, unlike a
    signal-peak heuristic, which resolves oppositely whenever the PCA
    fit window differs."""
    if rate_axis[1] < 0:
        return -rate_axis, -rate_pc1
    return rate_axis, rate_pc1


def fix_sign_acc(acc_axis, acc_pc1):
    """Canonical polarity for dynamic-accel PC1: the axis component
    with the largest magnitude is made positive (standard PCA sign
    convention, deterministic for a given axis)."""
    k = int(np.argmax(np.abs(acc_axis)))
    if acc_axis[k] < 0:
        return -acc_axis, -acc_pc1
    return acc_axis, acc_pc1


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def unwrap_deg(x_deg):
    """Unwrap an angle series given in degrees (handles +/-180 deg wraps)."""
    return np.rad2deg(np.unwrap(np.deg2rad(x_deg)))


# ---------------------------------------------------------------------------
# Loading / resampling
# ---------------------------------------------------------------------------
def load_capture(path):
    """Load the CSV and return (sample_idx, imu) with imu = (N, 6) columns
    roll, pitch, yaw, accel_x, accel_y, accel_z. Rows before the first IMU
    packet (all-NaN IMU cells) are dropped; isolated NaNs are interpolated."""
    raw = np.genfromtxt(path, delimiter=",", names=True)
    if raw.ndim == 0:
        raise SystemExit(f"No data rows found in {path}")
    names = raw.dtype.names or ()
    missing = [c for c in ("sample",) + IMU_COLS if c not in names]
    if missing:
        raise SystemExit(f"Missing columns in {path}: {missing}")

    sample = np.asarray(raw["sample"], dtype=float)
    imu = np.column_stack([np.asarray(raw[c], dtype=float) for c in IMU_COLS])

    valid = ~np.all(np.isnan(imu), axis=1)
    if not valid.any():
        raise SystemExit(f"No IMU rows found in {path}")
    sample, imu = sample[valid], imu[valid]

    for j in range(imu.shape[1]):  # fill any isolated gaps
        col = imu[:, j]
        bad = np.isnan(col)
        if bad.any() and (~bad).any():
            col[bad] = np.interp(sample[bad], sample[~bad], col[~bad])
    return sample, imu


def normalize_accel_units(imu, mode):
    """Convert legacy accel scaling to g if needed.

    Returns (imu, legacy, median_mag) so the caller can report the scale
    in the markdown capture table."""
    mag = np.median(np.linalg.norm(imu[:, 3:6], axis=1))
    if mode == "auto":
        legacy = mag > 8.0  # real data is ~1 g; legacy 1 g reads as ~256
    else:
        legacy = mode == "legacy"

    if legacy:
        print("=" * 74)
        print("DISCLAIMER: legacy accelerometer scaling detected "
              f"(median |accel| = {mag:.1f}).")
        print("This capture was recorded before the 2026-10-02 protocol.py fix")
        print("(raw/16 instead of raw/4096). Converting accel_x/y/z to g by 1/256.")
        print("Roll/pitch/yaw are unaffected. New captures are already in g.")
        print("=" * 74)
        imu = imu.copy()
        imu[:, 3:6] *= LEGACY_TO_G
    return imu, legacy, mag


def resample_uniform(sample, imu, emg_fs, imu_fs):
    """Resample the zero-order-held IMU streams onto a uniform imu_fs grid.
    Time is zero-based at the first IMU row. Returns (t, imu_resampled)."""
    t_raw = sample / emg_fs
    t_raw = t_raw - t_raw[0]
    n = int(round(t_raw[-1] * imu_fs)) + 1
    t = np.arange(n) / imu_fs
    out = np.column_stack([np.interp(t, t_raw, imu[:, j]) for j in range(imu.shape[1])])
    return t, out


# ---------------------------------------------------------------------------
# EMG envelope (same filter chain as plot_isometric_raw_modes.py)
# ---------------------------------------------------------------------------
ADS1299_VREF = 4.5


def compute_emg_envelope(path, channel, emg_fs, env_cutoff, gain):
    """Linear EMG envelope with the plot_isometric_raw_modes.py filter
    chain: 4th-order zero-phase bandpass 20-450 Hz (NO notch), rectify,
    4th-order zero-phase low-pass at ``env_cutoff`` Hz.

    Gaps in the sample-counter column are re-inserted as NaN and filtering
    runs per contiguous valid segment (>= 64 samples), exactly like
    plot_isometric_raw_modes.py. The envelope is converted to microvolts with
    Vref = 4.5 V and the given PGA gain.

    Returns (t_s, envelope_uv) over the FULL reconstructed timeline, with
    t in absolute sample time (sample_idx / emg_fs) so it aligns with the
    IMU traces (which are zero-order-held onto the same sample timeline).
    """
    if channel < 1:
        raise SystemExit("--channel must be >= 1")
    nyq = emg_fs / 2.0
    if nyq <= 400.0:
        raise SystemExit(f"EMG sample rate {emg_fs} Hz too low for a "
                         f"400 Hz bandpass (Nyquist {nyq:g} Hz)")
    if not 0.0 < env_cutoff < nyq:
        raise SystemExit(f"Envelope cutoff {env_cutoff} Hz must be between "
                         f"0 and Nyquist {nyq:g} Hz")

    df = pd.read_csv(path)
    cols = [c for c in df.columns
            if isinstance(c, str) and c.lower().startswith("ch")
            and c.lower()[2:].isdigit()]
    if not cols:
        raise SystemExit(f"No EMG channels (ch1, ch2, ...) found in {path}")
    ch_col = next((c for c in cols if c.lower() == f"ch{channel}"), None)
    if ch_col is None:
        raise SystemExit(f"Channel ch{channel} not found; "
                         f"available EMG columns: {cols}")

    idx = df.iloc[:, 0].values.astype(np.int64)
    x = df[ch_col].values.astype(float)

    # Gap reconstruction on the sample-counter column (as emg_isometric.py).
    full_start, full_end = idx[0], idx[-1]
    n_full = full_end - full_start + 1
    received = np.zeros(n_full, dtype=bool)
    received[idx - full_start] = True
    recon = np.full(n_full, np.nan)
    recon[received] = x
    n_missing = int((~received).sum())

    # Gap statistics, same content as emg_isometric.py, markdown format.
    d_idx = np.diff(idx)
    gap_positions = np.where(d_idx != 1)[0]
    gap_lengths = [int(d_idx[g] - 1) for g in gap_positions]
    print("\n## EMG sample loss\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| File | `{path}` |")
    print(f"| Channel | ch{channel} |")
    print(f"| Samples received | {int(received.sum())} |")
    print(f"| Samples missing | {n_missing} ({100.0 * n_missing / n_full:.2f}%) |")
    print(f"| Number of gaps | {len(gap_lengths)} |")
    if gap_lengths:
        gl = np.array(gap_lengths, dtype=float)
        print(f"| Gap length mean (samples) | {gl.mean():.1f} +/- {gl.std():.1f} (SD) |")
        print(f"| Gap length median (samples) | {np.median(gl):.0f} |")
        print(f"| Gap length min (samples) | {gl.min():.0f} |")
        print(f"| Gap length max (samples) | {gl.max():.0f} |")
        print(f"| Gap length mean (ms @{emg_fs:g} Hz) | "
              f"{gl.mean() / emg_fs * 1000:.1f} +/- {gl.std() / emg_fs * 1000:.1f} (SD) |")
        print(f"| Gap length max (ms @{emg_fs:g} Hz) | {gl.max() / emg_fs * 1000:.0f} |")
        print("\n### Gap details\n")
        print("| Start index | Length (samples) | Length (ms) |")
        print("|-------------|------------------|-------------|")
        for g, length in zip(gap_positions, gap_lengths):
            start_idx = int(idx[g])
            print(f"| {start_idx} | {length} | {length / emg_fs * 1000:.0f} |")

    # plot_isometric_raw_modes.py chain: 20-450 Hz bandpass, no notch.
    # (SENIAM says 20-500 Hz, but fs = 980 Hz gives Nyquist 490 Hz and a
    # Butterworth edge that close to Nyquist is numerically unstable;
    # 450 Hz keeps the design stable and covers the full EMG bandwidth.)
    b_bp, a_bp = signal.butter(4, [20.0 / nyq, 450.0 / nyq], btype="bandpass")
    b_env, a_env = signal.butter(4, env_cutoff / nyq, btype="lowpass")
    uv_per_code = ADS1299_VREF / gain / (2 ** 23) * 1e6
    print("\n## EMG envelope filters\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print("| Bandpass | 20-450 Hz, 4th order, zero-phase |")
    print("| Notch | none (mains stays in the signal) |")
    print(f"| Envelope low-pass | {env_cutoff:g} Hz, 4th order, zero-phase |")
    print(f"| PGA gain | {gain:g} |")
    print(f"| Scale | {uv_per_code:.4f} µV/code |")

    def valid_segments(mask, min_len=1):
        d = np.diff(mask.astype(int))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, len(mask)]
        return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]

    envelope = np.full(n_full, np.nan)
    for (s, e) in valid_segments(received, min_len=64):
        seg = signal.filtfilt(b_bp, a_bp, recon[s:e])
        envelope[s:e] = signal.filtfilt(b_env, a_env, np.abs(seg))

    t = np.arange(full_start, full_end + 1) / emg_fs
    return t, envelope * uv_per_code


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(t, imu, args):
    """Run the full sagittal-plane analysis. Returns a dict of signals/stats."""
    fs = args.imu_fs
    dt = 1.0 / fs

    # No IMU filters: raw (unwrapped) angles, mean-subtracted acceleration.
    roll_f = imu[:, 0]
    pitch_f = imu[:, 1]
    yaw_f = imu[:, 2]

    # Gravity estimate (mean accel): should be ~1 g for a static-ish sensor.
    g_vec = imu[:, 3:6].mean(axis=0)
    print("\n## Gravity estimate (mean accel)\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| accel_x | {g_vec[0]:+.3f} g |")
    print(f"| accel_y | {g_vec[1]:+.3f} g |")
    print(f"| accel_z | {g_vec[2]:+.3f} g |")
    print(f"| \\|g\\| | {np.linalg.norm(g_vec):.3f} g |")

    # Dynamic acceleration: subtracting the mean gravity vector removes
    # gravity + DC offset (no high-pass filter).
    acc_dyn = imu[:, 3:6] - g_vec

    # Angular rates from unwrapped Euler angles (deg/s).
    rate = np.column_stack([
        np.gradient(unwrap_deg(a), dt) for a in (roll_f, pitch_f, yaw_f)
    ])

    # Signal Vector Magnitude (SVM): orientation-independent motion intensity.
    # |a_dyn| needs no "-1 g" correction because gravity was already removed
    # by the high-pass (the -1 g ENMO form applies to raw accel magnitude).
    # NOTE: |omega| here is the magnitude of the *Euler rates*, a proxy for
    # the true body-frame |omega| (Euler rates mix axes with orientation);
    # still a valid intensity/cadence measure.
    rate_svm = np.linalg.norm(rate, axis=1)
    acc_svm = np.linalg.norm(acc_dyn, axis=1)

    # PCA fit window (default: whole file). Use --fit-start/--fit-end to
    # exclude standing still / turns from the PCA fit.
    fit = np.ones(t.size, dtype=bool)
    if args.fit_start is not None:
        fit &= t >= args.fit_start
    if args.fit_end is not None:
        fit &= t <= args.fit_end
    if fit.sum() < 10:
        raise SystemExit("PCA fit window has fewer than 10 samples")

    rate_vecs, rate_expl = pca(rate[fit])
    acc_vecs, acc_expl = pca(acc_dyn[fit])

    rate_axis = rate_vecs[:, 0]
    rate_pc1 = rate @ rate_axis
    rate_axis, rate_pc1 = fix_sign_rate(rate_axis, rate_pc1)

    acc_pc1 = acc_dyn @ acc_vecs[:, 0]
    acc_pc2 = acc_dyn @ acc_vecs[:, 1]
    acc_pc3 = acc_dyn @ acc_vecs[:, 2]
    acc_axis1, acc_pc1 = fix_sign_acc(acc_vecs[:, 0], acc_pc1)

    dom_rate = int(np.argmax(np.abs(rate_axis)))
    corr1 = np.corrcoef(rate_pc1, acc_pc1)[0, 1]
    corr2 = np.corrcoef(rate_pc1, acc_pc2)[0, 1]

    print("\n## Angular-rate PCA\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Fit window (samples) | {fit.sum()} |")
    print(f"| Fit window (s) | {t[fit][-1] - t[fit][0]:.1f} |")
    for k in range(3):
        print(f"| Explained variance PC{k + 1} | {rate_expl[k]:.3f} |")
    for k, name in enumerate(("roll", "pitch", "yaw")):
        print(f"| PC1 axis ({name}) | {rate_axis[k]:+.3f} |")
    print(f"| Dominant Euler axis | {ANGLE_NAMES[dom_rate]} |")
    print("(Sagittal-dominant walking: PC1 ~ pitch(y), explained PC1 > ~0.8.)")

    print("\n## Dynamic-accel PCA\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    for k in range(3):
        print(f"| Explained variance PC{k + 1} | {acc_expl[k]:.3f} |")
    for k, name in enumerate(("x", "y", "z")):
        print(f"| PC1 axis (a{name}) | {acc_axis1[k]:+.3f} |")
    print(f"| PC1+PC2 plane span | {acc_expl[0] + acc_expl[1]:.1%} |")
    print(f"| PC3 fraction | {acc_expl[2]:.1%} |")
    print("(A small PC3 fraction means mostly planar (sagittal) motion.)")

    print("\n## Rate-PC1 vs accel-PC correlation\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| rate-PC1 vs accel-PC1 | {corr1:+.3f} |")
    print(f"| rate-PC1 vs accel-PC2 | {corr2:+.3f} |")
    print("(Accel leads angular rate by ~90 deg in a sinusoidal sense, so a")
    print(" strong |corr| against either PC is expected; check the plot.)")

    print("\n## Signal Vector Magnitude (SVM)\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| \\|omega\\| mean | {rate_svm.mean():.1f} deg/s |")
    print(f"| \\|omega\\| max | {rate_svm.max():.1f} deg/s |")
    print(f"| \\|a_dyn\\| mean | {acc_svm.mean():.3f} g |")
    print(f"| \\|a_dyn\\| max | {acc_svm.max():.3f} g |")

    return {
        "t": t,
        "roll_f": roll_f, "pitch_f": pitch_f, "yaw_f": yaw_f,
        "rate": rate, "rate_pc1": rate_pc1, "rate_axis": rate_axis,
        "rate_expl": rate_expl,
        "acc_dyn": acc_dyn,
        "acc_pc1": acc_pc1, "acc_pc2": acc_pc2, "acc_pc3": acc_pc3,
        "acc_axis1": acc_axis1, "acc_expl": acc_expl,
        "rate_svm": rate_svm, "acc_svm": acc_svm,
        "fit": fit,
        "gravity": g_vec, "dom_rate": dom_rate,
        "corr1": corr1, "corr2": corr2,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_csv(path, res):
    header = ",".join([
        "t_s", "roll_deg", "pitch_deg", "yaw_deg",
        "droll_dps", "dpitch_dps", "dyaw_dps", "rate_pc1_dps",
        "acc_dyn_x_g", "acc_dyn_y_g", "acc_dyn_z_g",
        "acc_pc1_g", "acc_pc2_g", "acc_pc3_g",
        "rate_svm_dps", "acc_svm_g",
    ])
    data = np.column_stack([
        res["t"], res["roll_f"], res["pitch_f"], res["yaw_f"],
        res["rate"][:, 0], res["rate"][:, 1], res["rate"][:, 2], res["rate_pc1"],
        res["acc_dyn"][:, 0], res["acc_dyn"][:, 1], res["acc_dyn"][:, 2],
        res["acc_pc1"], res["acc_pc2"], res["acc_pc3"],
        res["rate_svm"], res["acc_svm"],
    ])
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    print(f"Saved processed signals to {path}")


# ---------------------------------------------------------------------------
# Per-trial pipeline CSVs (--save-trial-csv)
# ---------------------------------------------------------------------------
def _fmt(v):
    """Format one header-comment value: floats with 6 significant digits."""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _command_line():
    """The exact runnable command line (arguments quoted, like the logs)."""
    return "python " + " ".join(shlex.quote(a) for a in sys.argv[1:])


def write_trial_csvs(args, res, emg_t, emg_env, sample, legacy, accel_mag,
                     heel_strikes):
    """Write the two per-trial CSVs for the gait pipeline:

    <stem>_signals.csv - 50 Hz IMU/PCA grid: filtered angles, angular rates,
        rate PC1, dynamic accel, accel PC1/PC2/PC3, rate/accel SVM, with
        '# key: value' header comments carrying the PCA statistics, SVM
        statistics and the gait-cycle metadata.
    <stem>_emg.csv - EMG envelope on the --emg-fs grid (980 Hz) with
        '# key: value' header comments carrying the channel, gain, scale
        and the filter settings.

    Both files are written next to the capture CSV.
    """
    out_dir = os.path.dirname(os.path.abspath(args.input))
    base = os.path.splitext(os.path.basename(args.input))[0]

    # ---- <stem>_signals.csv --------------------------------------------
    t = res["t"]
    cycles = None
    durs = None
    if heel_strikes:
        cycles = list(zip(heel_strikes[:-1], heel_strikes[1:]))
        durs = np.array([b - a for a, b in cycles])

    hdr = [
        "# Gait trial signals (50 Hz IMU/PCA grid)",
        f"# command: {_command_line()}",
        f"# input: {args.input}",
        f"# imu_fs_hz: {args.imu_fs:g}",
        f"# emg_fs_hz: {args.emg_fs:g}",
        f"# imu_filters: none (raw angles, mean-subtracted accel)",
        f"# accel_unit: {'legacy->g (x 1/256)' if legacy else 'g'}",
        f"# median_abs_accel: {_fmt(accel_mag)}",
        f"# imu_rows: {sample.size}",
        f"# imu_t0_s: {_fmt(sample[0] / args.emg_fs)}",
        f"# duration_s: {_fmt(t[-1])}",
        f"# gravity_x_g: {_fmt(res['gravity'][0])}",
        f"# gravity_y_g: {_fmt(res['gravity'][1])}",
        f"# gravity_z_g: {_fmt(res['gravity'][2])}",
        f"# fit_start_s: {_fmt(args.fit_start) if args.fit_start is not None else 'none'}",
        f"# fit_end_s: {_fmt(args.fit_end) if args.fit_end is not None else 'none'}",
        f"# fit_samples: {int(res['fit'].sum())}",
        f"# rate_pca_explained_pc1: {_fmt(res['rate_expl'][0])}",
        f"# rate_pca_explained_pc2: {_fmt(res['rate_expl'][1])}",
        f"# rate_pca_explained_pc3: {_fmt(res['rate_expl'][2])}",
        f"# rate_pc1_axis_roll: {_fmt(res['rate_axis'][0])}",
        f"# rate_pc1_axis_pitch: {_fmt(res['rate_axis'][1])}",
        f"# rate_pc1_axis_yaw: {_fmt(res['rate_axis'][2])}",
        f"# rate_dominant_axis: {ANGLE_NAMES[res['dom_rate']]}",
        f"# acc_pca_explained_pc1: {_fmt(res['acc_expl'][0])}",
        f"# acc_pca_explained_pc2: {_fmt(res['acc_expl'][1])}",
        f"# acc_pca_explained_pc3: {_fmt(res['acc_expl'][2])}",
        f"# acc_pc1_axis_x: {_fmt(res['acc_axis1'][0])}",
        f"# acc_pc1_axis_y: {_fmt(res['acc_axis1'][1])}",
        f"# acc_pc1_axis_z: {_fmt(res['acc_axis1'][2])}",
        f"# acc_pc12_plane_span: {_fmt(res['acc_expl'][0] + res['acc_expl'][1])}",
        f"# rate_pc1_vs_acc_pc1_corr: {_fmt(res['corr1'])}",
        f"# rate_pc1_vs_acc_pc2_corr: {_fmt(res['corr2'])}",
        f"# rate_svm_mean_dps: {_fmt(res['rate_svm'].mean())}",
        f"# rate_svm_max_dps: {_fmt(res['rate_svm'].max())}",
        f"# acc_svm_mean_g: {_fmt(res['acc_svm'].mean())}",
        f"# acc_svm_max_g: {_fmt(res['acc_svm'].max())}",
    ]
    if cycles is not None:
        hdr.append(f"# heel_strikes_s: "
                   + ",".join(f"{x:g}" for x in heel_strikes))
        hdr.append(f"# n_cycles: {len(cycles)}")
        hdr.append(f"# cycle_duration_mean_s: {_fmt(durs.mean())}")
        hdr.append(f"# cycle_duration_sd_s: "
                   + _fmt(durs.std(ddof=1) if len(durs) > 1 else 0.0))
        hdr.append(f"# cycle_duration_min_s: {_fmt(durs.min())}")
        hdr.append(f"# cycle_duration_max_s: {_fmt(durs.max())}")
        hdr.append(f"# mean_cadence_cpm: "
                   + _fmt(60.0 * len(cycles) / durs.sum()))
        for k, (a, b) in enumerate(cycles, 1):
            hdr.append(f"# cycle_{k}_start_s: {_fmt(a)}")
            hdr.append(f"# cycle_{k}_end_s: {_fmt(b)}")
            hdr.append(f"# cycle_{k}_duration_s: {_fmt(b - a)}")

    header = "\n".join(hdr) + "\n"
    data = np.column_stack([
        t, res["roll_f"], res["pitch_f"], res["yaw_f"],
        res["rate"][:, 0], res["rate"][:, 1], res["rate"][:, 2],
        res["rate_pc1"],
        res["acc_dyn"][:, 0], res["acc_dyn"][:, 1], res["acc_dyn"][:, 2],
        res["acc_pc1"], res["acc_pc2"], res["acc_pc3"],
        res["rate_svm"], res["acc_svm"],
    ])
    col_header = ",".join([
        "t_s", "roll_deg", "pitch_deg", "yaw_deg",
        "droll_dps", "dpitch_dps", "dyaw_dps", "rate_pc1_dps",
        "acc_dyn_x_g", "acc_dyn_y_g", "acc_dyn_z_g",
        "acc_pc1_g", "acc_pc2_g", "acc_pc3_g",
        "rate_svm_dps", "acc_svm_g",
    ])
    sig_path = os.path.join(out_dir, base + "_signals.csv")
    with open(sig_path, "w") as f:
        f.write(header)
        f.write(col_header + "\n")
        np.savetxt(f, data, delimiter=",", fmt="%.8g")
    print(f"Saved trial signals to {sig_path}")

    # ---- <stem>_emg.csv -------------------------------------------------
    uv_per_code = ADS1299_VREF / args.gain / (2 ** 23) * 1e6
    env_hdr = [
        f"# Gait trial EMG envelope ({args.emg_fs:g} Hz EMG grid)",
        f"# command: {_command_line()}",
        f"# input: {args.input}",
        f"# channel: ch{args.channel}",
        f"# gain: {args.gain:g}",
        f"# uv_per_code: {uv_per_code:.6g}",
        f"# emg_fs_hz: {args.emg_fs:g}",
        f"# bandpass: 20-450 Hz, 4th order, zero-phase",
        f"# notch: none",
        f"# envelope_lowpass_hz: {args.env_cutoff:g}",
    ]
    env_path = os.path.join(out_dir, base + "_emg.csv")
    with open(env_path, "w") as f:
        f.write("\n".join(env_hdr) + "\n")
        f.write("t_s,emg_envelope_uV\n")
        np.savetxt(f, np.column_stack([emg_t, emg_env]),
                   delimiter=",", fmt="%.8g")
    print(f"Saved trial EMG envelope to {env_path}")


def make_pca_figure(res, args):
    """Build the 5-subplot PCA figure. Returns the figure, or None if
    matplotlib is unavailable."""
    if plt is None:
        print("matplotlib not available - skipping plot")
        return None
    t = res["t"]
    fig, axes = plt.subplots(5, 1, figsize=(22, 11), sharex=True,
                             constrained_layout=True)
    fig.suptitle(f"Sagittal-plane isolation: {args.input}")

    ax = axes[0]
    ax.plot(t, res["roll_f"], lw=0.8, label="roll")
    ax.plot(t, res["pitch_f"], lw=0.8, label="pitch")
    ax.plot(t, res["yaw_f"], lw=0.8, label="yaw")
    ax.set_ylabel("angle (deg)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[1]
    for j, name in enumerate("xyz"):
        ax.plot(t, res["acc_dyn"][:, j], lw=0.5, alpha=0.5, label=f"dyn a{name}")
    ax.plot(t, res["acc_pc1"], lw=1.2, color="k", label="accel PC1")
    ax.plot(t, res["acc_pc2"], lw=1.2, color="tab:red", label="accel PC2")
    ax.set_ylabel("dyn accel (g)")
    ax.legend(loc="upper right", fontsize=8, ncol=5)

    ax = axes[2]
    for j, name in enumerate(ANGLE_NAMES):
        ax.plot(t, res["rate"][:, j], lw=0.5, alpha=0.5, label=f"d{name}")
    ax.plot(t, res["rate_pc1"], lw=1.2, color="k", label="rate PC1 (sagittal)")
    ax.set_ylabel("ang. rate (deg/s)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)

    ax = axes[3]
    ax.plot(t, res["rate_svm"], lw=0.9, color="tab:blue",
            label="|omega| (rate SVM)")
    ax.set_ylabel("|omega| (deg/s)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[4]
    ax.plot(t, res["acc_svm"], lw=0.9, color="tab:green",
            label="|a_dyn| (accel SVM)")
    ax.set_ylabel("|a_dyn| (g)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)

    return fig


def make_combined_figure(res, emg_t, emg_env, t_imu0, args):
    """Build the 3-subplot EMG/IMU figure: EMG envelope, dynamic-accel PC1,
    angular-rate PC1 on one shared time axis. ``t_imu0`` shifts the IMU time
    (zero-based at the first IMU row) into the EMG sample-time base."""
    if plt is None:
        return None
    t_imu = res["t"] + t_imu0
    fig, axes = plt.subplots(3, 1, figsize=(22, 8), sharex=True,
                             constrained_layout=True)
    fig.suptitle(f"EMG envelope vs sagittal-plane PCs: {args.input}")

    ax = axes[0]
    ax.plot(emg_t, emg_env, lw=0.8, color="tab:blue",
            label=f"EMG ch{args.channel} envelope")
    ax.set_ylabel("EMG envelope (µV)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_imu, res["acc_pc1"], lw=0.9, color="k", label="accel PC1")
    ax.set_ylabel("dyn accel PC1 (g)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t_imu, res["rate_pc1"], lw=0.9, color="k",
            label="rate PC1 (sagittal)")
    ax.set_ylabel("ang. rate PC1 (deg/s)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig


def parse_heel_strikes(spec):
    """Parse '10.5,12.3,14.1' into a sorted list of heel-strike times (s).
    At least 2 times are required: they bound one gait cycle."""
    try:
        strikes = sorted(float(x) for x in spec.split(",") if x.strip())
    except ValueError:
        raise SystemExit(f"Could not parse --heel-strikes '{spec}' as a "
                         f"comma-separated list of times in seconds")
    if len(strikes) < 2:
        raise SystemExit("--heel-strikes needs at least 2 times "
                         "(start and end of one gait cycle)")
    return strikes


def make_gait_cycle_figure(res, emg_t, emg_env, t_imu0, heel_strikes, args):
    """Build the 3-subplot gait-cycle overlay figure: EMG envelope,
    dynamic-accel PC1 and angular-rate PC1, each overlaid across gait cycles
    (individual traces + mean + 90 % CI, like the contraction overlays in
    emg_isometric.py). Consecutive heel strikes bound one gait cycle, so
    n strikes give n-1 cycles. Cycles are time-normalized to 0-100 percent
    (default) or kept on a real-time axis from heel strike (--time-axis time;
    the mean then stops at the shortest cycle, as in emg_isometric.py)."""
    if plt is None:
        return None
    t_imu = res["t"] + t_imu0

    cycles = list(zip(heel_strikes[:-1], heel_strikes[1:]))
    durs = np.array([b - a for a, b in cycles])
    if (durs <= 0).any():
        raise SystemExit("--heel-strikes contains duplicate/non-increasing times")
    dur_sd = durs.std(ddof=1) if len(durs) > 1 else 0.0
    print("\n## Gait cycles\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Heel strikes | {len(heel_strikes)} |")
    print(f"| Gait cycles | {len(cycles)} |")
    print(f"| Duration mean | {durs.mean():.3f} s |")
    print(f"| Duration SD | {dur_sd:.3f} s |")
    print(f"| Duration min | {durs.min():.3f} s |")
    print(f"| Duration max | {durs.max():.3f} s |")
    print(f"| Mean cadence | {60.0 * len(cycles) / durs.sum():.1f} cycles/min |")
    print("\n### Cycle details\n")
    print("| Cycle | Start (s) | End (s) | Duration (s) |")
    print("|-------|-----------|---------|--------------|")
    for k, (a, b) in enumerate(cycles, 1):
        print(f"| {k} | {a:.2f} | {b:.2f} | {b - a:.3f} |")

    t_lo = min(emg_t[0], t_imu[0])
    t_hi = max(emg_t[-1], t_imu[-1])
    outside = [h for h in heel_strikes if h < t_lo or h > t_hi]
    if outside:
        print(f"WARNING: {len(outside)} heel strike(s) outside the data range "
              f"[{t_lo:.1f}, {t_hi:.1f}] s: "
              + ", ".join(f"{h:.2f}" for h in outside))

    if args.time_axis == "percent":
        grid = np.linspace(0.0, 100.0, 201)
        x_label = "% of gait cycle (0 = heel strike)"
    else:
        grid = np.arange(0.0, durs.min(), 0.01)
        x_label = "Time from heel strike (s)"

    signals = [
        (emg_t, emg_env, f"EMG ch{args.channel} envelope", "µV"),
        (t_imu, res["acc_pc1"], "dyn accel PC1", "g"),
        (t_imu, res["rate_pc1"], "ang. rate PC1", "deg/s"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             constrained_layout=True)
    fig.suptitle(f"Gait-cycle overlays ({len(cycles)} cycles): {args.input}")

    for ax, (t_sig, y_sig, label, unit) in zip(axes, signals):
        curves = []
        for (a, b) in cycles:
            if args.time_axis == "percent":
                x_cycle = (t_sig - a) / (b - a) * 100.0
            else:
                x_cycle = t_sig - a
            m = (~np.isnan(y_sig)) & (t_sig >= a) & (t_sig <= b)
            if m.sum() < 10:
                continue
            curves.append(np.interp(grid, x_cycle[m], y_sig[m]))
        if not curves:
            ax.text(0.5, 0.5, "No valid cycles in this signal", ha="center",
                    transform=ax.transAxes)
            continue
        curves = np.stack(curves)
        n = curves.shape[0]
        for k in range(n):
            ax.plot(grid, curves[k], color="0.85", lw=0.7, ls="--", alpha=0.6,
                    label="Individual" if k == 0 else None)
        mean = curves.mean(axis=0)
        if n > 1:
            ci = 1.645 * curves.std(axis=0, ddof=1) / np.sqrt(n)
            ax.fill_between(grid, mean - ci, mean + ci, color="0.35",
                            alpha=0.35, lw=0, label="90% CI")
        ax.plot(grid, mean, color="black", lw=2.0, label=f"Mean (n={n})")
        ax.set_ylabel(f"{label} ({unit})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="PCA-based sagittal-plane isolation from a Sokosti capture CSV "
                    "(roll/pitch/yaw + accel). See module docstring for the "
                    "legacy accelerometer-scaling disclaimer.",
    )
    p.add_argument("input", help="Capture CSV with columns: sample, roll, pitch, "
                                 "yaw, accel_x, accel_y, accel_z")
    p.add_argument("--emg-fs", type=float, default=980.0,
                   help="CSV row rate in Hz used for the time base. The "
                        "Walk captures were recorded at 980 sps (not "
                        "1000); all time windows on the command line must "
                        "be given in this time base (default: 980)")
    p.add_argument("--imu-fs", type=float, default=50.0,
                   help="IMU update rate in Hz to resample to (default: 50)")
    p.add_argument("--lp-fc", type=float, default=10.0,
                   help="Low-pass cutoff for angles in Hz (default: 10)")
    p.add_argument("--hp-fc", type=float, default=0.2,
                   help="High-pass cutoff for gravity removal in Hz (default: 0.2)")
    p.add_argument("--order", type=int, default=4,
                   help="Butterworth filter order (default: 4)")
    p.add_argument("--accel-unit", choices=("auto", "g", "legacy"), default="auto",
                   help="Accelerometer scaling of the input: auto-detect, already "
                        "in g, or legacy pre-2026-10-02 scale (default: auto)")
    p.add_argument("--fit-start", type=float, default=None,
                   help="PCA fit window start in s (exclude standing/turns)")
    p.add_argument("--fit-end", type=float, default=None,
                   help="PCA fit window end in s")
    p.add_argument("--save-csv", default=None,
                   help="Optional path to save the processed signals")
    p.add_argument("--save-plot", default=None,
                   help="Optional path to save the plot as an image")
    p.add_argument("--no-plot", action="store_true",
                   help="Do not open an interactive plot window")
    p.add_argument("--channel", type=int, default=1,
                   help="1-based EMG channel used for the envelope in the "
                        "combined EMG/IMU figure (default: 1)")
    p.add_argument("--env-cutoff", type=float, default=6.0,
                   help="Low-pass cutoff (Hz) for the EMG linear envelope, "
                        "same default as emg_isometric.py (default: 6)")
    p.add_argument("--gain", type=float, default=8.0,
                   help="ADS1299 PGA gain used during recording, for "
                        "converting the envelope to microvolts "
                        "(Vref = 4.5 V; default 8, the live-capture default)")
    p.add_argument("--heel-strikes", default=None,
                   help='Comma-separated heel-strike times in seconds, e.g. '
                        '"10.5,12.3,14.1". Consecutive strikes bound one gait '
                        'cycle (n strikes -> n-1 cycles) for the gait-cycle '
                        'overlay figure')
    p.add_argument("--time-axis", choices=("percent", "time"), default="percent",
                   help="X axis of the gait-cycle overlays: normalize each "
                        "cycle to 0-100 percent (default) or real seconds "
                        "from heel strike")
    p.add_argument("--save-results", action="store_true",
                   help="Save ALL figures and a terminal-output log file "
                        "(<csv>_imu_analysis.log) into the CSV's folder, "
                        "like emg_isometric.py --save-results")
    p.add_argument("--save-trial-csv", action="store_true",
                   help="Write the per-trial pipeline CSVs next to the "
                        "capture CSV: <stem>_signals.csv (50 Hz IMU/PCA grid "
                        "with header-comment statistics) and <stem>_emg.csv "
                        "(EMG envelope on the --emg-fs grid), consumed by "
                        "plot_gait_trials.py / plot_gait_group.py")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # --save-results: tee all terminal output into a buffer (like
    # emg_isometric.py); flushed to <csv>_imu_analysis.log at the end.
    _log_buf = None
    _log_ctx = None
    if args.save_results:
        import contextlib
        import io
        import shlex
        _log_buf = io.StringIO()
        _log_ctx = contextlib.redirect_stdout(_log_buf)
        _log_ctx.__enter__()
        _log_buf.write("# Command: python "
                       + " ".join(shlex.quote(a) for a in sys.argv[1:]) + "\n\n")

    sample, imu = load_capture(args.input)
    imu, legacy, accel_mag = normalize_accel_units(imu, args.accel_unit)
    t, imu = resample_uniform(sample, imu, args.emg_fs, args.imu_fs)

    print("\n## Capture\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| File | `{args.input}` |")
    print(f"| IMU-bearing rows | {sample.size} |")
    scale_txt = ("legacy -> converted to g (x 1/256)" if legacy
                 else "g (already correct)")
    print(f"| Accelerometer scale | {scale_txt} |")
    print(f"| Median \\|accel\\| | {accel_mag:.3f} |")
    print(f"| Resampled IMU rate | {args.imu_fs:g} Hz |")
    print(f"| Resampled samples | {t.size} |")
    print(f"| Duration | {t[-1]:.1f} s |")

    res = analyze(t, imu, args)

    # EMG envelope over the full CSV timeline (raw-modes filter chain).
    emg_t, emg_env = compute_emg_envelope(
        args.input, args.channel, args.emg_fs, args.env_cutoff, args.gain)

    # The IMU signals are zero-based at the first IMU row; shift them into
    # the EMG sample-time base so both share one x axis.
    t_imu0 = sample[0] / args.emg_fs

    if args.save_csv:
        save_csv(args.save_csv, res)

    fig_pca = make_pca_figure(res, args)
    fig_comb = make_combined_figure(res, emg_t, emg_env, t_imu0, args)

    fig_gait = None
    heel_strikes = None
    if args.heel_strikes:
        heel_strikes = parse_heel_strikes(args.heel_strikes)
        fig_gait = make_gait_cycle_figure(res, emg_t, emg_env, t_imu0,
                                          heel_strikes, args)

    if args.save_trial_csv:
        write_trial_csvs(args, res, emg_t, emg_env, sample, legacy,
                         accel_mag, heel_strikes)

    if args.save_plot and fig_pca is not None:
        fig_pca.savefig(args.save_plot, dpi=150)
        print(f"Saved plot to `{args.save_plot}`")

    if args.save_results:
        out_dir = os.path.dirname(os.path.abspath(args.input))
        base = os.path.splitext(os.path.basename(args.input))[0]
        png_pca = os.path.join(out_dir, base + "_sagittal_pca.png")
        png_comb = os.path.join(out_dir, base + "_emg_imu.png")
        png_gait = os.path.join(out_dir, base + "_gait_cycles.png")
        log_path = os.path.join(out_dir, base + "_imu_analysis.log")
        if fig_pca is not None:
            fig_pca.savefig(png_pca, dpi=150)
        if fig_comb is not None:
            fig_comb.savefig(png_comb, dpi=150)
        if fig_gait is not None:
            fig_gait.savefig(png_gait, dpi=150)
        if _log_ctx is not None:
            _log_ctx.__exit__(None, None, None)
        with open(log_path, "w") as f:
            f.write(_log_buf.getvalue())
        print("\n## Saved output\n")
        print("| File | Path |")
        print("|------|------|")
        print(f"| PCA figure | `{png_pca}` |")
        print(f"| EMG/IMU figure | `{png_comb}` |")
        if fig_gait is not None:
            print(f"| Gait-cycle figure | `{png_gait}` |")
        print(f"| Log file | `{log_path}` |")

    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()

