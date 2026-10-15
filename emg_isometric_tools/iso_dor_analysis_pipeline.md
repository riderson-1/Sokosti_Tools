# Iso_Dor Analysis Pipeline Documentation

This document describes the complete analysis pipeline for the isometric
contraction (Iso_Dor) EMG trials: from the raw CSV recordings to the final
SNR plots. It covers the files involved, how data flows between them, and
the decisions made along the way.

---

## Overview

```
Raw CSV recordings          emg_isometric.py            parse_emg_logs.py         plot_isometric_trials.py
(BLE/USB capture)     -->   per-trial analysis    -->   log parser          -->   SNR plots
                            + *_analysis.log            + emg_isometric_summary.ods
                            + PNG figures               (4 tidy sheets)

                         plot_isometric_raw_modes.py  (Stage 4, reads raw CSVs + ODS)
                              --> raw SNR / MDF / FFT figures
```

| Stage | File | Location |
|-------|------|----------|
| 1. Analysis | `emg_isometric.py` | `/home/karl/repos/Sokosti_tools/` |
| 2. Batch runner | `rerun_iso_dor.sh` | `/home/karl/repos/Sokosti_tools/` |
| 3. Log parser | `parse_emg_logs.py` | `/home/karl/Documents/Master_Thesis/Testing/application_testing/` |
| 4. Plotting | `plot_isometric_trials.py` | `/home/karl/repos/Sokosti_tools/` |
| 5. Raw-mode plots | `plot_isometric_raw_modes.py` | `/home/karl/repos/Sokosti_tools/emg_isometric_tools/` |

Data locations:

- Raw CSVs and per-trial logs:
  `/home/karl/Documents/Master_Thesis/Testing/application_testing/EMG_Testing_<date>/<trial>/`
- Summary workbook:
  `/home/karl/Documents/Master_Thesis/Testing/application_testing/emg_isometric_summary.ods`
- Plots: written to the current working directory (or `--out-dir`).

There are **12 Iso_Dor trials**: 6 recording sessions (2 subjects × 3 dates)
× 2 sources each (PC + SD variant). The PC recording is the primary analysis
target; the SD variant is analyzed with the same pipeline.

---

## Stage 0: Raw data (CSV files)

Each trial folder contains one CSV per recording source, e.g.

```
EMG_Testing_08092026/BLE_1_Karl_Iso_Dor_2026-09-08/BLE_1_Karl_Iso_Dor_2026-09-08_PC.csv
```

The CSVs are written by the live capture tools (`emg_live_ble.py` /
`emg_live_plot.py`) and contain **raw ADS1299 ADC codes**, not volts:

```
sample,status1_ok,status2_ok,ch1,...,ch16,roll,pitch,yaw,accel_x,accel_y,accel_z
223056,1,1,307316,290660,...
```

- `sample` — device sample counter (used for gap/lost-sample detection)
- `ch1..ch16` — 24-bit signed ADC codes from the ADS1299
- IMU columns (roll/pitch/yaw/accel) — not used in the isometric analysis

Key properties of the raw codes:

- Large DC offset (~300,000 codes ≈ 0.17 V electrode offset) — normal,
  removed by the bandpass filter.
- Sample rate **fs = 980 Hz** (recordings are 980 sps, not 1000 — verified
  from the 50 Hz mains peak; all time windows are scaled by 1000/980 so they
  select the same samples as the old fs = 1000 analysis).
- BLE recordings contain gaps (dropped samples) visible in the `sample`
  counter; USB recordings are nearly gap-free.

---

## Stage 1: Per-trial analysis — `emg_isometric.py`

Run per trial (batched by `rerun_iso_dor.sh`, see Stage 1b):

```
.venv/bin/python emg_isometric.py <csv> --channels <ch> --thresh-strong <t> \
    --gain <g> --compare-channels --psd fft --show-windows \
    --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50 \
    --fs 980 --save-results
```

### Processing chain

1. **Load CSV** and reconstruct the full sample timeline from the `sample`
   counter. Missing samples are inserted as NaN (gap-aware processing).
2. **Gap statistics** — count, mean/median/min/max length (samples and ms),
   and a per-gap detail table. Printed and logged.
3. **Filtering** (per contiguous valid segment, so gaps don't smear):
   - Bandpass **20–400 Hz** (4th-order Butterworth) — removes the DC offset
     and out-of-band noise.
   - Notch **48–52 Hz** — removes mains interference.
4. **Linear envelope** — rectify + 6 Hz low-pass Butterworth.
5. **Contraction windows** — manual windows given on the command line
   (`--strong-windows` / `--weak-windows`), determined visually per trial
   from the magnitude plot (all 8 BLE runs). The 4 USB runs use
   auto-detection; its duration thresholds (`--merge-gap 0.511`,
   `--strong-min 2.0405`, `--weak-min 10.2035`) are the old
   0.5 s / 2 s / 10 s scaled by 1000/980, so detection selects the same
   samples as at fs = 1000.
6. **Channel comparison** (`--compare-channels`) — for *every* channel:
   - **Signal** = mean envelope (µV) over the contraction window, trimmed
     symmetrically by `--signal-trim` (5 %) each side to exclude ramp-up/down.
   - **Noise** = mean envelope (µV) in a window ending `--noise-offset`
     (1.530612 s = 1500 samples at 980 Hz) before the contraction onset,
     spanning `--noise-window` (50 % of the contraction duration).
   - **SNR** = 20·log10(signal/noise), per contraction and per channel.
   - **MDF** (median frequency) from the FFT of the filtered EMG.
7. **Unit conversion** — all reported values are in **microvolts (µV)**:
   `VOLTS_PER_CODE = Vref / gain / 2^23` (Vref = 4.5 V), then × 1e6.
   The `--unit` option was removed; µV is hardcoded everywhere.

### Decisions

- **Gain matters**: the ADS1299 PGA gain (1 or 8, per recording) must be
  passed via `--gain`; it scales every absolute value. SNR (a ratio) is
  gain-independent.
- **Noise window before onset, not after**: the pre-contraction rest is used
  so the noise estimate is not contaminated by the contraction itself.
- **Signal trim 5 %**: excludes the ramp-up/ramp-down of each contraction so
  the signal reflects the plateau.
- **Manual windows**: BLE contraction onsets/ends were set by hand per trial
  (auto-detection thresholds were unreliable across subjects/dates); the
  USB trials use the (sample-scaled) auto-detection thresholds.
- **FFT over full contiguous segments**: the whole recording (gap-aware) is
  used for the MDF spectrum.

### Outputs (per trial, written next to the CSV)

- `<stem>_<ch>_magnitude.png` — filtered EMG + envelope, all channels
- `<stem>_<ch>_overlays.png` — magnitude + PSD + strong/weak overlays
- **`<stem>_analysis.log`** — the full terminal output, starting with the
  exact runnable command (`# Command: python ...` with quoted window lists),
  so any analysis can be reproduced verbatim.

### Stage 1b: Batch runner — `rerun_iso_dor.sh`

Contains the 12 corrected commands (one per trial/source). It exists because
earlier log files had broken command lines (missing `python` prefix, unquoted
window lists, flag typos like `--strong-window`). Rerunning it regenerates
all logs and figures with the current script version.

---

## Stage 2: Log parser — `parse_emg_logs.py`

```
.venv/bin/python parse_emg_logs.py            # writes emg_isometric_summary.ods
```

Walks `application_testing/` for `*_analysis.log` files inside any
`*Iso_Dor*` trial folder and parses each log **by locating markdown tables
via their header cells** (no fixed positions, tolerant to layout changes).

### Extracted data → 4 tidy sheets in `emg_isometric_summary.ods`

| Sheet | Grain | Contents |
|-------|-------|----------|
| `trials` | 1 row per trial (12) | subject/date/connection/source, run settings parsed from the `# Command:` line (gain, channels, thresholds, trim, noise params, psd), samples received/missing, gap stats, contraction counts, best/worst channel |
| `mdf` | trial × channel (192) | per-channel mean signal (µV), mean noise (µV), SNR, SNR dB, MDF, n |
| `snr` | trial × channel × contraction (1568) | window start/end, contraction type (strong/weak), signal (µV), noise (µV), SNR, SNR dB |
| `gaps` | trial × gap event (3029) | gap start index, length in samples and ms |

All signal/noise columns carry the `_uV` suffix — values are in µV.

### Decisions

- **Parse from logs, not Excel**: the previous workflow pasted console output
  into an ODS by hand; this parser reads the logs directly so the workbook
  can always be regenerated.
- **Trial metadata from the file name** (`USB_1_Karl_Iso_Dor_2026-09-02_PC`)
  and **run settings from the logged command line** — no manual bookkeeping.
- **Contraction type** (strong/weak) is matched to each detail-table row via
  the ordered "Contractions detected" list (strong first, then weak).

---

## Stage 3: Plotting — `plot_isometric_trials.py`

```
.venv/bin/python plot_isometric_trials.py <path-to-emg_isometric_summary.ods>
```

Produces `snr_strong.(png|pdf)`, `snr_weak.(png|pdf)` and
`snr_plot_data.csv` (exactly the numbers plotted).

### Method

1. **PC recordings only** (`SOURCE = "PC"`); SD variants excluded for now.
2. **Noise floor per channel** = mean of `noise_uV` over the **weak rests
   2–4** (chronological rank). The **first weak rest is excluded**
   (`NOISE_SKIP_FIRST_WEAK = 1`) because it directly follows the maximal
   strong block — noise there is 10–20× higher in several trials. Rests 2–4
   follow submaximal contractions and represent the true baseline.
3. **Channel selection per trial**: every channel gets
   `score = mean(strong SNR dB, weak SNR dB)` where both SNRs are
   **recalculated against the 2–4 noise floor** (not the per-contraction
   `snr_db` from the sheet). The channel with the highest score is used for
   both plots. (Note: this can differ from ranking by the sheet's own
   `snr_db` — e.g. BLE_8_Max selects Ch3, not Ch14.)
4. **Signal values** = `signal_uV` of every strong (or weak) contraction of
   the selected channel; the box spans min–max, the diamond is the mean.
5. **SNR (dB)** = `20·log10(mean signal / 2–4 noise floor)`, one red dot per
   trial on the right axis, vertically aligned with the box and noise bar.
6. Two figures: strong contractions and weak contractions, shared y-limits
   (log µV axis) unless `--separate-ylim`.

### Decisions

- **Common noise floor for strong and weak plots**: the same 2–4 rest noise
  is used in both, so the two figures are directly comparable.
- **Skip the first weak rest**: documented above; the excluded rest's noise
  is printed in the console summary (`excl. rest` column) for transparency.
- **One channel per trial** (best combined SNR) rather than per-plot
  channels, so strong and weak refer to the same electrode.
- **Individual contraction dots removed** from the boxes — box (min–max) +
  mean diamond is enough visually.

---

## Stage 4: SENIAM-bandpass mode comparison plots — `plot_isometric_raw_modes.py`

```
.venv/bin/python emg_isometric_tools/plot_isometric_raw_modes.py \
    emg_isometric_tools/emg_isometric_summary.ods --out-dir emg_isometric_tools
```

### Filtering: SENIAM 20–450 Hz bandpass, no notch

**All numbers from this stage are computed after:**

1. per-channel DC (mean) removal, then
2. a **20–450 Hz 4th-order Butterworth bandpass**, applied with `filtfilt`
   (zero-phase) per contiguous gap-free segment.

**No 50 Hz notch is applied** — mains interference stays in the signal by
design. The only other filter is the 6 Hz low-pass of the linear envelope
(rectify + LPF) that produces the signal/noise amplitudes.

The SENIAM standard bandpass is 20–500 Hz, but the recordings are
**fs = 980 Hz (Nyquist = 490 Hz)** and a Butterworth band edge that close to
Nyquist is numerically **unstable** (verified: filter poles with |z| > 1 at a
490 Hz upper edge, which corrupts the output — MDF slammed to 490 Hz). The
upper edge is therefore **450 Hz**: a stable design (pole check enforced in
code) that still covers the full EMG bandwidth. This differs from Stages 1–3
(20–400 Hz + 48–52 Hz notch), so the two sets of numbers are not directly
comparable to each other.

The script reads **both** inputs:

- `emg_isometric_summary.ods` — contraction windows (manual BLE / auto USB
  windows, already verified in Stage 1), rest ranks, run settings
  (gain, fs, signal-trim, noise-offset, noise-window) and the PC trial list.
  Windows are **not re-detected**.
- The **raw CSVs** (located next to each trial's `log_path`) — all signal
  values are recomputed from scratch here, then filtered as above.

### Three figures (modes: USB gain 1, BLE gain 1, BLE gain 8; PC only)

1. **`snr_raw_modes.(png|pdf)`** — ONE box plot, 3 boxes. One data point =
   one contraction, **strong AND weak pooled**, both subjects; the 16 channels
   are averaged into ONE signal value per contraction:
   - *signal* = mean over the 16 channels of the mean envelope in the
     contraction window (trimmed `--signal-trim` = 5 % each side);
   - *noise* = per channel, mean envelope in the rest window ending
     `--noise-offset` (1.530612 s) before onset and spanning `--noise-window`
     (50 % of contraction duration); averaged over **weak rests 2–4** (the
     first weak rest is skipped — it follows the maximal strong block), then
     averaged over the 16 channels → one floor per trial;
   - **SNR = 20·log10(signal / noise)** in dB, one value per contraction.
2. **`mdf_raw_modes.(png|pdf)`** — ONE box plot, same 3 boxes. Per-channel
   MDF over the whole recording (gap-aware segment-averaged FFT of the
   bandpassed signal, median frequency of the magnitude² spectrum), pooled
   over all 16 channels × 2 subjects per mode (n = 32 each).
3. **`fft_raw_modes.(png|pdf)`** — one spectrum curve per mode with a
   dashed vertical line at that curve's MDF. Per mode, all channels of all
   recordings are concatenated into one long vector (gap samples removed;
   per-channel mean subtracted so boundary steps stay small) and **ONE single
   FFT** is computed over the whole vector — no Welch/chunk averaging. The
   magnitude (|rfft|/N, µV) is **max-pooled onto the fs/nfft grid**
   (nfft = 8192 → ~0.12 Hz bins — the same grid `emg_isometric.py` uses for
   its FFT figure) and plotted on a **linear axis stopping at 0** (classic
   FFT look; x-limit 0–500 Hz, same as `emg_isometric.py`). Max-pooling keeps
   narrow peaks such as the 50 Hz mains line; the MDF line itself is computed
   from the full-resolution spectrum (pooling is display-only).

### Outputs (written to `--out-dir`)

- `snr_raw_modes.png|pdf`, `mdf_raw_modes.png|pdf`, `fft_raw_modes.png|pdf`
- `snr_raw_modes_plot_data.csv` — per-contraction SNR rows (mode, trial,
  subject, type, window, signal µV, noise floor µV, SNR dB)
- `mdf_raw_modes_plot_data.csv` — per-channel MDF rows
- `fft_raw_modes_plot_data.csv` — the plotted spectra on the same ~0.12 Hz
  grid as the figure (mode, f_hz, magnitude_uV, mdf_hz; 4096 rows per mode).
- A console summary table with all 3 SNR and 3 MDF numbers.

### Verified behaviour (20–450 Hz, no notch)

| Mode | SNR mean (dB) | MDF box mean (Hz) | MDF pooled-curve (Hz) |
|------|--------------:|-------------------:|----------------------:|
| USB gain 1 | 26.1 | 78.5 | 78.8 |
| BLE gain 1 | 27.1 | 70.6 | 63.9 |
| BLE gain 8 | 33.4 | 67.4 | 59.9 |

- MDF values (60–79 Hz) are physiologically typical for surface EMG.
- The 50 Hz mains peak is **present** in the spectra (+3–4 dB above the local
  floor) — expected, since no notch is applied.
- Stopbands verified: < 20 Hz and > 450 Hz strongly attenuated.

---



```bash
cd /home/karl/repos/Sokosti_tools

# 1. per-trial analyses (regenerates logs + PNGs next to the CSVs)
bash rerun_iso_dor.sh

# 2. parse logs -> ODS workbook
.venv/bin/python ../Documents/Master_Thesis/Testing/application_testing/parse_emg_logs.py

# 3. plots
.venv/bin/python plot_isometric_trials.py \
    /home/karl/Documents/Master_Thesis/Testing/application_testing/emg_isometric_summary.ods

# 4. raw-mode plots (raw CSVs + ODS; no bandpass/notch - see Stage 4)
.venv/bin/python emg_isometric_tools/plot_isometric_raw_modes.py \
    emg_isometric_tools/emg_isometric_summary.ods --out-dir emg_isometric_tools
```

Python environment: the workspace venv (`.venv/bin/python`, Python 3.12) has
numpy/scipy/pandas/matplotlib/odfpy installed.

---

## Known caveats

- **BLE gap rates vary widely** between trials (0 to >1500 gaps); the gap
  statistics in the `trials` sheet quantify this. Filtering is gap-aware,
  but long gaps still remove data.
- **Manual contraction windows** are subjective; they are recorded in each
  log's `# Command:` line so the exact windows can always be audited.
- **The first weak rest is excluded from the noise floor** by design; if a
  trial has fewer than 4 weak contractions the floor uses fewer rests.
- **SD variants** are analyzed but currently excluded from the plots
  (`SOURCE = "PC"`).
