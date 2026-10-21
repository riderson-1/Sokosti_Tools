#!/usr/bin/env python3
"""
Pooled SNR / MDF / FFT plots for the PC isometric trials, on data filtered
according to the SENIAM guideline.

FILTERING - every number here is computed after:
  1. per-channel DC (mean) removal, then
  2. a 20-450 Hz 4th-order Butterworth bandpass applied with filtfilt
     (zero-phase) per contiguous gap-free segment.
SENIAM specifies a 20-500 Hz standard bandpass, but the recordings are
fs = 980 Hz (Nyquist = 490 Hz) and a Butterworth band edge that close to
Nyquist is numerically unstable (verified: poles with |z| > 1 at 490 Hz,
which corrupts the output). The upper edge is therefore 450 Hz - a stable
design that still covers the entire EMG bandwidth.
NO 50 Hz NOTCH is applied - mains interference stays in the signal by design.
The only other filter is the 6 Hz low-pass of the linear envelope
(rectify + LPF) that turns the EMG into signal/noise amplitudes.

Three figures, one per connection/gain combination:
  1. snr_raw_modes - ONE box plot with 3 boxes (USB gain 1, BLE gain 1,
     BLE gain 8). One data point = one contraction in one recording, strong
     AND weak pooled, all subjects; the 16 channels are averaged into ONE
     signal value per contraction. The displayed n counts contraction-level
     values, not channels or recordings:
        signal = mean over the 16 channels of the mean envelope in the
                 contraction window (trimmed --signal-trim each side)
        noise  = the trial's rest 2-4 floor: per channel the mean
                 envelope in the pre-contraction rest window (ending
                 --noise-offset before onset, spanning --noise-window % of
                 the contraction), averaged over weak rests 2-4 (the FIRST
                 weak rest is skipped: it follows the maximal strong block),
                 then averaged over the 16 channels
        SNR    = 20*log10(signal/noise) in dB
  2. mdf_raw_modes - ONE box plot with the same 3 boxes: one MDF per channel
     per recording, calculated over the whole recording (gap-aware
     segment-averaged FFT of the bandpassed signal, median frequency of the
     magnitude^2 spectrum). The displayed n counts recording-channel pairs
     (16 channels per recording); e.g. 2 recordings x 16 channels = 32
     possible values.
  3. fft_raw_modes - one spectrum curve per mode with a vertical line at that
     curve's MDF. Per mode, all channels of all recordings are concatenated
     into one long vector (gap samples removed; per-channel mean subtracted
     so boundary steps stay small) and ONE single FFT is computed over the
     whole vector - no Welch/chunk averaging. The magnitude (|rfft|/N, uV) is
     max-pooled onto the fs/nfft grid (nfft = 8192 -> ~0.12 Hz bins), the same
     grid emg_isometric.py uses for its FFT figure, and plotted on a LINEAR
     axis stopping at 0 (classic FFT look; max-pooling keeps narrow peaks
     such as the 50 Hz mains line). Its legend reports N, the total sample
     count in that concatenated vector, and N/fs, the equivalent pooled
     channel-data duration (not wall-clock recording time).

Contraction windows, rest ranks and run settings come from
emg_isometric_summary.ods (sheets 'snr' and 'trials'); the SIGNALS themselves
are recomputed from the raw CSVs that sit next to each trial's log file,
then filtered as described above.

Input : emg_isometric_summary.ods
Output: snr_raw_modes.(png|pdf), mdf_raw_modes.(png|pdf),
        fft_raw_modes.(png|pdf), snr_raw_modes_plot_data.csv,
        mdf_raw_modes_plot_data.csv, fft_raw_modes_plot_data.csv
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
from matplotlib.patches import Rectangle
from scipy import signal as sps

# ===== SETTINGS =====
VREF = 4.5                 # ADS1299 reference voltage (V)
ENV_CUTOFF = 6.0           # envelope low-pass cutoff (Hz)
BP_LOW = 20.0              # SENIAM bandpass lower edge (Hz)
BP_HIGH = 450.0            # bandpass upper edge (Hz). SENIAM says 20-500 Hz,
                           # but fs = 980 Hz gives Nyquist = 490 Hz and a
                           # Butterworth edge that close to Nyquist is
                           # numerically UNSTABLE (poles |z| > 1, verified);
                           # 450 Hz keeps the design stable and still covers
                           # the full EMG bandwidth.
NOISE_SKIP_FIRST_WEAK = 1  # leading weak rest(s) excluded from the noise floor
NFFT_FFT = 8192            # nfft grid: per-channel MDF box values AND the
                           # displayed/pooled FFT curve bins (fs/8192 ~ 0.12 Hz
                           # at fs 980) - same grid as emg_isometric.py
BOX_WIDTH = 0.32
SUBJECTS = ["Max", "Karl"]
SIGNAL_COLOR = "lightblue"
SIGNAL_MEAN_COLOR = "royalblue"
MDF_COLOR = "lightgreen"
MDF_MEAN_COLOR = "seagreen"
MODE_COLORS = ["tab:blue", "tab:green", "tab:red"]

REQUIRED_TRIAL_COLS = ["trial_id", "subject", "source", "connection", "gain",
                       "fs_hz", "signal_trim", "noise_offset_s",
                       "noise_window_pct", "log_path"]
REQUIRED_SNR_COLS = ["trial_id", "source", "channel", "contraction_idx",
                     "contraction_type", "window_start_s", "window_end_s"]

# ===== ARGUMENTS =====
parser = argparse.ArgumentParser(
    description="SENIAM 20-450 Hz bandpass (no notch) pooled SNR / MDF / "
                "FFT plots for the PC isometric trials.")
parser.add_argument("ods", help="Path to the summary workbook "
                                "(e.g. emg_isometric_summary.ods)")
parser.add_argument("--out-dir", default=None,
                    help="Folder for the output files (default: same folder "
                         "as the summary ODS file)")
parser.add_argument("--no-show", action="store_true",
                    help="Save the figures without opening windows.")
args = parser.parse_args()


# ===== HELPERS (gap-aware, same definitions as emg_isometric.py) =====
def valid_segments(mask, min_len=1):
    """Return (start, end) index pairs of contiguous True runs."""
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def fft_segment_average(x, fs, min_seg=512, nfft=8192):
    """Average FFT magnitude over contiguous non-NaN segments of length >= min_seg
    (identical to emg_isometric.py: full-length FFT per segment, interpolated
    onto a common nfft grid, weighted by segment length)."""
    valid = ~np.isnan(x)
    f_grid = np.fft.rfftfreq(nfft, 1 / fs)
    mags, weights = [], []
    for (s, e) in valid_segments(valid, min_len=min_seg):
        seg = sps.detrend(x[s:e])
        f_seg = np.fft.rfftfreq(len(seg), 1 / fs)
        mags.append(np.interp(f_grid, f_seg, np.abs(np.fft.rfft(seg))))
        weights.append(e - s)
    if not mags:
        return None, None
    return f_grid, np.average(np.stack(mags), axis=0, weights=weights)


def median_frequency(f, power):
    """MDF: the frequency below which 50 % of the total power lies."""
    cum = np.cumsum(power)
    if cum[-1] <= 0:
        return np.nan
    return float(f[np.searchsorted(cum, 0.5 * cum[-1])])


def win_mean(win):
    """Mean of a window's non-NaN samples; NaN if fewer than 10 samples."""
    v = win[~np.isnan(win)]
    return float(np.mean(v)) if len(v) >= 10 else np.nan


# ===== LOAD =====
def load(path):
    for sheet, cols in (("trials", REQUIRED_TRIAL_COLS), ("snr", REQUIRED_SNR_COLS)):
        df = pd.read_excel(path, sheet_name=sheet)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing columns in sheet '{sheet}': {missing}")
        if sheet == "trials":
            trials = df
        else:
            snr = df
    bad = snr[~snr["contraction_type"].isin(["strong", "weak"])]
    if len(bad):
        raise SystemExit("contraction_type invalid for trials: "
                         + ", ".join(sorted(bad["trial_id"].unique())))
    return trials, snr


def build_groups(trials, snr):
    """Group the PC trials into the three (connection, gain) modes."""
    pc = trials[trials["source"] == "PC"]
    if pc.empty:
        raise SystemExit("No rows with source == 'PC' in sheet 'trials'.")
    groups = []
    for (connection, gain), g in pc.groupby(["connection", "gain"], sort=False):
        mode_trials = []
        for subject in SUBJECTS:
            tids = sorted(g.loc[g["subject"] == subject, "trial_id"].unique())
            if not tids:
                raise SystemExit(f"No PC trial for subject '{subject}' in "
                                 f"{connection} gain {gain}.")
            mode_trials.extend(tids)
        groups.append({"label": f"{connection} gain {int(gain)}",
                       "connection": connection, "gain": int(gain),
                       "trial_ids": mode_trials})
    # stable, readable order: USB first, then BLE gain 1, then BLE gain 8
    groups.sort(key=lambda d: (d["connection"] != "USB", d["connection"], d["gain"]))
    if len(groups) != 3:
        raise SystemExit(f"Expected exactly 3 (connection, gain) modes, got "
                         f"{len(groups)}: {[g['label'] for g in groups]}")

    # contraction windows: identical across channels -> one row per contraction
    wins = snr[(snr["source"] == "PC")
               & snr["trial_id"].isin([t for g in groups for t in g["trial_ids"]])]
    wins = wins.dropna(subset=["window_start_s", "window_end_s"])
    chk = wins.groupby(["trial_id", "contraction_idx"])["window_start_s"].nunique()
    if (chk > 1).any():
        raise SystemExit("Window start differs between channels for: "
                         + ", ".join(chk[chk > 1].index.get_level_values(0)))
    wins = (wins.drop_duplicates(["trial_id", "contraction_idx"])
            [["trial_id", "contraction_idx", "contraction_type",
              "window_start_s", "window_end_s"]]
            .sort_values(["trial_id", "contraction_idx"]))
    return groups, wins, pc.set_index("trial_id")


# ===== PER-TRIAL RAW ANALYSIS =====
def load_raw(csv_path, gain):
    """CSV -> gap-reconstructed raw codes (n x 16) and the received mask."""
    df = pd.read_csv(csv_path)
    chans = sorted([c for c in df.columns
                    if isinstance(c, str) and c.lower().startswith("ch")
                    and c.lower()[2:].isdigit()],
                   key=lambda c: int(c[2:]))
    if len(chans) != 16:
        raise SystemExit(f"{csv_path}: expected 16 channels, found {len(chans)}")
    idx = df["sample"].to_numpy(np.int64)
    mask = np.zeros(idx[-1] - idx[0] + 1, dtype=bool)
    mask[idx - idx[0]] = True
    recon = np.full((len(mask), len(chans)), np.nan)
    recon[mask] = df[chans].to_numpy(float)
    # codes -> uV; per-channel DC removal before the bandpass (the 20 Hz
    # highpass removes DC anyway, but this keeps filtfilt headroom small)
    scale = VREF / gain / (2 ** 23) * 1e6
    recon = (recon - np.nanmean(recon, axis=0)) * scale
    return recon, mask


def raw_envelope(recon, mask, fs):
    """Rectify + 6 Hz low-pass per contiguous segment (envelope of the
    bandpassed signal)."""
    b_env, a_env = sps.butter(4, ENV_CUTOFF / (fs / 2), btype="lowpass")
    env = np.full_like(recon, np.nan)
    for (s, e) in valid_segments(mask, min_len=64):
        for ci in range(recon.shape[1]):
            env[s:e, ci] = sps.filtfilt(b_env, a_env, np.abs(recon[s:e, ci]))
    return env


def bandpass(recon, mask, fs):
    """SENIAM 20-450 Hz 4th-order Butterworth, filtfilt (zero-phase), applied
    per contiguous gap-free segment. SENIAM specifies 20-500 Hz, but the
    recordings are fs = 980 Hz (Nyquist 490 Hz) and a Butterworth band edge
    that close to Nyquist is numerically unstable (verified: poles |z| > 1
    at 490 Hz), so the upper edge is 450 Hz - stable, and the passband still
    covers the whole EMG spectrum. No notch filter is applied."""
    nyq = fs / 2.0
    high = min(BP_HIGH, nyq * (1.0 - 1e-6))  # never design at/above Nyquist
    b_bp, a_bp = sps.butter(4, [BP_LOW / nyq, high / nyq], btype="bandpass")
    poles = np.max(np.abs(np.roots(a_bp)))
    if poles >= 1.0:  # should never trigger; guards against bad parameters
        raise SystemExit(f"Bandpass design unstable (max|pole| = {poles:.6f})")
    out = np.full_like(recon, np.nan)
    for (s, e) in valid_segments(mask, min_len=64):
        for ci in range(recon.shape[1]):
            out[s:e, ci] = sps.filtfilt(b_bp, a_bp, recon[s:e, ci])
    return out


def contraction_windows(trial_id, wins, meta, env):
    """Per-contraction (idx, type, start, end, signal, noise, snr) from the
    bandpassed envelopes; noise rests 2-4 pooled into one floor per trial."""
    trim = float(meta["signal_trim"])
    offset = float(meta["noise_offset_s"])
    pct = float(meta["noise_window_pct"])
    fs = float(meta["fs_hz"])

    rows = wins[wins["trial_id"] == trial_id]
    weak = rows[rows["contraction_type"] == "weak"].sort_values("window_start_s")
    rest_rows = weak.iloc[NOISE_SKIP_FIRST_WEAK:]  # skip the first weak rest

    # per-channel rest 2-4 floor: mean over the used rests, then over channels
    rest_means = []
    for _, r in rest_rows.iterrows():
        a, b = r["window_start_s"], r["window_end_s"]
        d = b - a
        nwin = (pct / 100.0) * d
        i0 = max(0, int((a - offset - nwin) * fs))
        i1 = max(0, int((a - offset) * fs))
        rest_means.append([win_mean(env[i0:i1, ci])
                           for ci in range(env.shape[1])])
    floor_ch = np.nanmean(np.asarray(rest_means), axis=0)  # per channel
    floor = np.nanmean(floor_ch)                           # channel-set floor
    if not np.isfinite(floor) or floor <= 0:
        raise SystemExit(f"{trial_id}: invalid noise floor {floor}")

    out = []
    for _, r in rows.iterrows():
        a, b = r["window_start_s"], r["window_end_s"]
        d = b - a
        i0 = int((a + trim * d) * fs)
        i1 = int((b - trim * d) * fs)
        sig_ch = np.array([win_mean(env[i0:i1, ci])
                           for ci in range(env.shape[1])])
        sig = np.nanmean(sig_ch)
        snr_db = 20.0 * np.log10(sig / floor) if sig > 0 else np.nan
        out.append({"trial_id": trial_id,
                    "contraction_idx": int(r["contraction_idx"]),
                    "contraction_type": r["contraction_type"],
                    "window_start_s": a, "window_end_s": b,
                    "signal_uV": sig, "noise_floor_uV": floor,
                    "snr_db": snr_db})
    return out


def accumulate_vector(vec, acc):
    """Append one channel's gap-free vector to the mode's concatenated signal.
    The channel mean is subtracted first so the channel-to-channel boundary
    steps in the concatenated vector stay small (the bandpass already removed
    DC; this is belt-and-braces)."""
    v = vec[~np.isnan(vec)]
    if v.size == 0:
        return 0
    acc["parts"].append(v - v.mean())
    return v.size


def single_fft_spectrum(acc, fs):
    """ONE FFT over the mode's full concatenated vector (no Welch/chunk
    averaging). Returns (f_hz, magnitude, mdf_hz, sample_count).

    The magnitude (|rfft| / N, so units stay uV and modes of slightly
    different lengths stay comparable) is max-pooled onto the same
    fs/nfft grid that emg_isometric.py uses for its FFT plot (nfft = 8192
    -> ~0.12 Hz bins at fs = 980), which gives the classic, uncrowded
    FFT look. Max-pooling (instead of point sampling) guarantees narrow
    peaks such as the 50 Hz mains line survive the coarser binning.
    """
    v = np.concatenate(acc["parts"])
    n = v.size
    mag = np.abs(np.fft.rfft(v)) / n
    f = np.fft.rfftfreq(n, 1 / fs)
    # MDF from the FULL-RESOLUTION spectrum (pooling would bias it);
    # pooling is display-only
    mdf = median_frequency(f, mag ** 2)
    fc, mc = max_pool(f, mag, df=fs / NFFT_FFT)
    return fc, mc, mdf, n


def max_pool(f, power, df=0.05):
    """Max-pool the full-resolution spectrum onto a df-Hz grid (peaks such as
    mains survive pooling). Returns (f_center, pooled_power)."""
    cell = np.floor(f / df).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(cell)) + 1]
    pooled = np.maximum.reduceat(power, starts)
    centers = (cell[starts] + 0.5) * df
    return centers, pooled


# ===== PLOT =====
def draw_box(ax, x, vals, facecolor, mean_color, fmt="{:.1f}"):
    """Box-and-whisker at x: whiskers = min..max, box = Q1..Q3, line = median,
    diamond = mean with its value (style of the existing allch plots)."""
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    lo, hi, mean = vals.min(), vals.max(), vals.mean()
    ax.vlines(x, lo, q1, color="black", linewidth=1.2, zorder=2)
    ax.vlines(x, q3, hi, color="black", linewidth=1.2, zorder=2)
    for y in (lo, hi):
        ax.hlines(y, x - BOX_WIDTH / 4, x + BOX_WIDTH / 4,
                  color="black", linewidth=1.2, zorder=2)
    ax.add_patch(Rectangle((x - BOX_WIDTH / 2, q1), BOX_WIDTH, q3 - q1,
                           facecolor=facecolor, edgecolor="black",
                           alpha=0.8, linewidth=1.2, zorder=3))
    ax.hlines(med, x - BOX_WIDTH / 2, x + BOX_WIDTH / 2,
              color="black", linewidth=2.0, zorder=4)
    ax.scatter(x, mean, marker="D", s=70, facecolor=mean_color,
               edgecolor="black", zorder=5)
    ax.annotate(fmt.format(mean), (x, mean), textcoords="offset points",
                xytext=(-45, -1), ha="right", va="center",
                fontsize=13, color=mean_color, fontweight="bold")


def mode_xticklabels(groups, ns):
    return [f"{g['connection']} Gain {int(g['gain'])},\nn = {n}"
            for g, n in zip(groups, ns)]


def plot_box_figure(groups, values, y_label, title, out_stem, facecolor,
                    mean_color, fmt, out_dir, show):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    centres = np.arange(1, len(groups) + 1)
    for xc, vals in zip(centres, values):
        draw_box(ax, xc, vals, facecolor, mean_color, fmt=fmt)
    ax.set_xlim(centres[0] - 0.75, centres[-1] + 0.75)
    ax.set_xticks(centres)
    ax.set_xticklabels(mode_xticklabels(groups, [len(v) for v in values]),
                       fontsize=13)
    ax.set_ylabel(y_label, fontsize=15)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis="y", labelsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{out_stem}.{ext}"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def format_duration(seconds):
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.1f} s"


def plot_fft_figure(groups, freqs, mags, mdfs, sample_counts, fs, out_dir, show):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for g, f, mag, mdf, n_samples, color in zip(
            groups, freqs, mags, mdfs, sample_counts, MODE_COLORS):
        pooled_duration = format_duration(n_samples / fs)
        ax.plot(f, mag, color=color, linewidth=0.8,
                label=f"{g['label']} (MDF {mdf:.1f} Hz)\n"
                      f"N = {n_samples:,} samples; "
                      f"pooled recording data = {pooled_duration}")
        ax.axvline(mdf, color=color, linestyle="--", linewidth=1.5, alpha=0.85)
    # Single legend entry for all MDF dashed lines
    mdf_proxy = mlines.Line2D([], [], color="grey", linestyle="--",
                              linewidth=1.5, alpha=0.85,
                              label="Median frequency (MDF)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mdf_proxy)
    ax.set_xlim(0, 500)                 # same x-limit as emg_isometric.py
    ax.set_ylim(bottom=0)               # linear magnitude stops at 0
    ax.set_xlabel("Frequency (Hz)", fontsize=15)
    ax.set_ylabel("FFT magnitude (µV)", fontsize=15)
    ax.set_title("EMG Spectra",
                 fontsize=13)
    ax.legend(
        handles=handles, fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fft_raw_modes.{ext}"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ===== MAIN =====
def main():
    trials, snr = load(args.ods)
    groups, wins, trial_meta = build_groups(trials, snr)
    out_dir = args.out_dir if args.out_dir is not None \
        else os.path.dirname(os.path.abspath(args.ods))
    os.makedirs(out_dir, exist_ok=True)

    snr_rows, mdf_rows = [], []
    fft_acc = {g["label"]: {"parts": []} for g in groups}
    fs_global = None

    for g in groups:
        for trial_id in g["trial_ids"]:
            meta = trial_meta.loc[trial_id]
            fs = float(meta["fs_hz"])
            fs_global = fs
            gain = int(meta["gain"])
            csv_path = os.path.join(os.path.dirname(str(meta["log_path"])),
                                    f"{trial_id}.csv")
            if not os.path.exists(csv_path):
                raise SystemExit(f"Raw CSV not found: {csv_path}")
            print(f"[{g['label']}] {trial_id}: reading {os.path.basename(csv_path)} "
                  f"(gain {gain}, fs {fs:g})")
            recon, mask = load_raw(csv_path, gain)
            # SENIAM 20-450 Hz bandpass (no notch) -> everything below uses this
            bp = bandpass(recon, mask, fs)
            del recon
            env = raw_envelope(bp, mask, fs)

            # --- SNR: one value per contraction (16 channels averaged) ---
            for row in contraction_windows(trial_id, wins, meta, env):
                row["mode"] = g["label"]
                row["subject"] = meta["subject"]
                snr_rows.append(row)

            # --- MDF: one value per channel over the whole recording ---
            for ci in range(bp.shape[1]):
                f, mag = fft_segment_average(bp[:, ci], fs, nfft=NFFT_FFT)
                mdf_rows.append({"mode": g["label"], "trial_id": trial_id,
                                 "channel": f"Ch{ci + 1}",
                                 "mdf_hz": median_frequency(f, mag ** 2)
                                 if f is not None else np.nan})

            # --- FFT: gap-free per-channel vectors concatenated per mode;
            #         ONE single FFT over the whole vector (no chunk averaging) ---
            acc = fft_acc[g["label"]]
            for ci in range(bp.shape[1]):
                accumulate_vector(bp[:, ci], acc)

            del bp, mask, env

    snr_df = pd.DataFrame(snr_rows)
    mdf_df = pd.DataFrame(mdf_rows)

    # ===== figures =====
    snr_values = [snr_df.loc[snr_df["mode"] == g["label"], "snr_db"].dropna().to_numpy()
                  for g in groups]
    mdf_values = [mdf_df.loc[mdf_df["mode"] == g["label"], "mdf_hz"].dropna().to_numpy()
                  for g in groups]
    if any(len(v) == 0 for v in snr_values + mdf_values):
        raise SystemExit("Empty SNR/MDF group - check the mode grouping.")

    plot_box_figure(
        groups, snr_values, "SNR (dB)",
        "Signal-to-Noise Ratio",
        "snr_raw_modes", SIGNAL_COLOR, SIGNAL_MEAN_COLOR, "{:.1f}",
        out_dir, not args.no_show)
    plot_box_figure(
        groups, mdf_values, "MDF (Hz)",
        "Median Frequency",
        "mdf_raw_modes", MDF_COLOR, MDF_MEAN_COLOR, "{:.1f}",
        out_dir, not args.no_show)

    freqs, mags, mdfs, fft_sample_counts = [], [], [], []
    for g in groups:
        acc = fft_acc[g["label"]]
        if not acc["parts"]:
            raise SystemExit(f"No FFT data for mode {g['label']}")
        f, mag, mdf, n_samples = single_fft_spectrum(acc, fs_global)
        freqs.append(f)
        mags.append(mag)
        mdfs.append(mdf)
        fft_sample_counts.append(n_samples)
    plot_fft_figure(groups, freqs, mags, mdfs, fft_sample_counts, fs_global,
                    out_dir, not args.no_show)

    # ===== plot data CSVs =====
    snr_df.to_csv(os.path.join(out_dir, "snr_raw_modes_plot_data.csv"),
                  index=False)
    mdf_df.to_csv(os.path.join(out_dir, "mdf_raw_modes_plot_data.csv"),
                  index=False)
    # FFT plot data: the plotted, binned spectrum (fs/nfft ~ 0.12 Hz grid -
    # identical to what the figure shows, ~4100 rows per mode), plus the
    # raw concatenated sample count and equivalent channel-data duration.
    fft_rows = []
    for g, f, mag, mdf, n_samples in zip(
            groups, freqs, mags, mdfs, fft_sample_counts):
        fft_rows.append(pd.DataFrame({"mode": g["label"], "f_hz": f,
                                      "magnitude_uV": mag, "mdf_hz": mdf,
                                      "n_samples": n_samples,
                                      "pooled_channel_data_duration_s":
                                          n_samples / fs_global}))
    fft_df = pd.concat(fft_rows, ignore_index=True)
    fft_df.to_csv(os.path.join(out_dir, "fft_raw_modes_plot_data.csv"),
                  index=False)

    # ===== console summary =====
    print("\n### Mode summary (SENIAM 20-450 Hz bandpass, no notch)\n")
    print("| Mode | SNR n | SNR mean (dB) | SNR median (dB) | MDF box mean (Hz) "
          "| MDF pooled-curve (Hz) |")
    print("|------|------:|--------------:|----------------:|-------------------:"
          "|---------------------:|")
    for i, g in enumerate(groups):
        s = snr_values[i]
        m = mdf_values[i]
        print(f"| {g['label']} | {len(s)} | {s.mean():.1f} | {np.median(s):.1f} "
              f"| {m.mean():.1f} | {mdfs[i]:.1f} |")
    print("\nOutputs written to", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
