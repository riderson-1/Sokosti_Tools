# Gait Analysis Pipeline Documentation

This document describes the complete analysis pipeline for the walking
(Fast/Slow/Incline Walk) EMG/IMU trials: from the raw CSV recordings to the
gait-cycle-locked overlay plots. It mirrors the Sit_Stand pipeline
(`sit_stand_tools/sit_stand_pipeline.md`), but the timed events are
**heel strikes**: consecutive strikes bound one gait cycle (n strikes →
n−1 cycles; the first strike starts the cycle, the next strike ends it and
starts the new one).

---

## Overview

```
Raw CSV recordings          gait_analysis.py              rerun_gait.sh             plot_gait_trials.py
(BLE/USB capture)     -->   per-trial analysis      -->   batch runner, 36    -->   gait-cycle-locked
                            + *_imu_analysis.log          runs (logged heel         overlay plots
                            + PNG figures                 strike times)             + plot-data CSV
                            + <stem>_signals.csv
                            + <stem>_emg.csv
                                                          plot_gait_group.py
                                                      --> group figures (gain-1 PC
                                                          trials, 2x2 quadrants,
                                                          one per walk type)
                                                          + gait_plot_data.csv
```

| Stage | File | Location |
|-------|------|----------|
| 1. Analysis | `gait_analysis.py` | `/home/karl/repos/Sokosti_tools/imu_gait_tools/` |
| 1b. Heel-strike picking | `pick_heel_strikes.py` | `/home/karl/repos/Sokosti_tools/imu_gait_tools/` |
| 2. Batch runner | `rerun_gait.sh` | `/home/karl/repos/Sokosti_tools/imu_gait_tools/` |
| 3. Plotting | `plot_gait_trials.py` | `/home/karl/repos/Sokosti_tools/imu_gait_tools/` |
| 3b. Group plotting | `plot_gait_group.py` | `/home/karl/repos/Sokosti_tools/imu_gait_tools/` |

Data locations:

- Raw CSVs, per-trial logs, PNGs and per-trial CSVs:
  `/home/karl/Documents/Master_Thesis/Testing/application_testing/EMG_Testing_<date>/<trial>/`
- Plots + plot-data CSV: written into `imu_gait_tools/` (or `--out-dir`).

There are **18 Walk trials**: 6 recording sessions (2 subjects × 3 dates,
USB_1 / BLE_1 / BLE_8) × 3 walk types each (Fast, Slow, Incline), each
analyzed for both the PC and SD capture variants → **36 runs** in the batch
runner.

---

## Stage 1: Per-trial analysis — `gait_analysis.py`

Run per trial (batched by `rerun_gait.sh`, see Stage 2):

```
.venv/bin/python gait_analysis.py <csv> --channel <ch> --env-cutoff 15 \
    --gain <g> --heel-strikes '<t1,t2,...>' --emg-fs 980 \
    --time-axis percent --no-plot --save-results --save-trial-csv
```

**Sample rate is 980 sps, not 1000** (same as the Iso_Dor and Sit_Stand
trials): `--emg-fs` defaults to 980, and every command-line time
(`--heel-strikes`, `--fit-start`/`--fit-end`) lives in that time base.
Times recorded against the old fs=1000 axis must be scaled by 1000/980 to
select the same samples (see Stage 2).

The processing chain (load → resample → filter → PCA → SVM → canonical sign →
figures) is documented in the module docstring of `gait_analysis.py`. The
pipeline-relevant addition is `--save-trial-csv`, which writes **two CSVs
per trial** next to the capture CSV.

**Per-trial settings** (reproduced verbatim from the existing
`*_imu_analysis.log` `# Command:` lines):

- **EMG channel varies per subject/session**: Karl ch16, Max-USB ch2,
  Max-BLE ch3.
- **`--env-cutoff 15`** (not the script default of 6) for all walk trials.
- **Gain**: 1 for the USB_1 and BLE_1 sessions, 8 (the script default) for
  the BLE_8 sessions.
- **Heel strikes**: 11 strikes per trial → 10 gait cycles (the original
  manual picks, taken from the logs; the Stage 1b picker replaces them
  with the full set of strikes in the recording — n strikes → n−1 cycles).

### `<stem>_signals.csv` — 50 Hz IMU/PCA grid

One row per resampled IMU sample:

```
t_s, roll_deg, pitch_deg, yaw_deg, droll_dps, dpitch_dps, dyaw_dps,
rate_pc1_dps, acc_dyn_x_g, acc_dyn_y_g, acc_dyn_z_g, acc_pc1_g, acc_pc2_g,
acc_pc3_g, rate_svm_dps, acc_svm_g
```

`# key: value` header comments carry everything the plotter (and any audit)
needs:

- `command` — the exact runnable command line (quoted, like the Iso_Dor logs)
- capture info: `imu_fs_hz`, `emg_fs_hz`, filter cutoffs, `accel_unit`,
  `imu_rows`, `duration_s`, `imu_t0_s` (IMU time offset in the EMG
  sample-time base — the IMU time axis is zero-based at the first IMU row)
- PCA statistics: explained variance PC1–PC3 (rate + accel), PC1 axis
  components, dominant axis, PC1+PC2 plane span, rate↔accel correlations
- SVM statistics: `|omega|` and `|a_dyn|` mean/max
- cycle metadata: `heel_strikes_s`, `n_cycles`, duration mean/SD/min/max,
  `mean_cadence_cpm`, and per-cycle `cycle_<k>_start_s` /
  `cycle_<k>_end_s` / `cycle_<k>_duration_s`

### `<stem>_emg.csv` — 980 Hz EMG envelope

One row per EMG sample: `t_s, emg_envelope_uV`. Header comments carry the
command line, channel, gain, µV/code scale and the filter settings
(bandpass 20–400 Hz, notch 48–52 Hz, envelope low-pass).

### Decisions

- **Two CSVs instead of one**: the IMU/PCA signals live on the 50 Hz
  resampled grid, the EMG envelope on the 980 Hz EMG grid; merging them
  would NaN-fill 95 % of the IMU columns or downsample the EMG.
- **Statistics as header comments**: a CSV is a single table, so the
  PCA/cycle metadata go into `# key: value` comment lines at the top —
  auditable, machine-readable, and ignored by `pandas.read_csv(..., comment="#")`.
- **8-significant-digit CSV formatting**: the trial CSVs are written with
  `%.8g` (not `%.6g`). With `%.6g`, an absolute EMG timestamp like
  1114.816 s was rounded to 1114.82 — 10 ms resolution — so 10 consecutive
  1 ms samples collapsed onto one timestamp and the cycle-percent
  interpolation produced a staircase (visibly "choppy") overlay. Whether a
  trial was affected depended on the magnitude of its start time
  (t ≳ 1000 s loses the 1 ms digit; t < 1000 s keeps it).
- **Heel strikes: picked `heel_strikes.txt` wins, else the log-derived
  hardcoded times**: all 18 trials were originally analyzed with 11 manual
  heel-strike times; the batch runner reuses them from each log's
  `# Command:` line, **scaled by 1000/980** so they select the same
  samples on the 980 sps time base (same convention as
  `rerun_iso_dor.sh` and `rerun_sit_stand.sh`). Since Stage 1b
  (`pick_heel_strikes.py`) a trial's `heel_strikes.txt` overrides the
  hardcoded times for that trial — but only when its first line contains
  `pick_heel_strikes.py`. The **old hand-written** `heel_strikes.txt`
  files (incomplete — 3 of 18 missing — and partly stale, e.g.
  BLE_8_Karl_Slow_Walk_2026-09-14 starts at 1025 in the txt but 1026.1 in
  the log) lack that header and are ignored, both by the picker's preload
  and by `rerun_gait.sh`, so they can never silently change results.
- **`imu_t0_s`**: the IMU time axis is zero-based at the first IMU row while
  the EMG time axis is absolute (sample/fs). The offset is stored so the
  plotter can align both onto one timeline.
- **`gravity`/`dom_rate`/`corr1`/`corr2` added to `analyze()`'s return
  dict**: the gait `analyze()` computed these for the terminal tables but
  did not return them; the trial-CSV writer needs them for the header
  comments (the sit-stand `analyze()` already returned them).

---

## Stage 1b: Heel-strike picking — `pick_heel_strikes.py`

Interactive replacement for typing strike times by hand: it opens two
x-aligned panels of a trial's **`*_PC_signals.csv`** — dynamic-accel
PC1/PC2 on top, sagittal angular-rate PC1 below — (the PC recordings,
fewer sample losses), lets you mark *every* heel strike in the full
recording on either panel, and writes `<trial>/heel_strikes.txt`, which
`rerun_gait.sh` prefers over its hardcoded times.

```bash
/home/karl/repos/Sokosti_tools/.venv/bin/python \
    /home/karl/repos/Sokosti_tools-wt/gait-heel-strike-picker/imu_gait_tools/pick_heel_strikes.py \
    /home/karl/Documents/Master_Thesis/Testing/application_testing/EMG_Testing_<date>/<trial>/<trial>_PC_signals.csv
```

Run it from the branch's **git worktree**
(`/home/karl/repos/Sokosti_tools-wt/gait-heel-strike-picker`), which sits
beside the `main` checkout (`/home/karl/repos/Sokosti_tools`) so both
branches can be open at the same time without fighting over one working
tree. The worktree has no `.venv`, so always use the main workspace's
interpreter (as above); `rerun_gait.sh` resolves it automatically
(override with `VENV=/path/to/python`).

Mouse / keys:

- **left click** add a strike (snapped to the nearest EMG sample, on
  either panel) ·
  **right click** remove the nearest mark (within 5 % of the view) ·
  **u** undo · **c** clear · **s** save & close · **q** quit without saving
- the toolbar magnifier zooms (drag a rectangle) and the arrow cross pans;
  while either tool is toggled on, marking clicks are ignored — toggle the
  tool off again to add/remove strikes (so a pan click can never drop a
  strike, and toolbar drag actions never add marks).

Key properties:

- The x-axis is the absolute EMG-sample/980 base (`t_s + imu_t0_s`) —
  exactly what `--heel-strikes` expects, so picked times need **no
  1000/980 scaling** (unlike the hardcoded log times).
- Times are absolute, so one picked list applies to **both** the PC and
  SD runs of the trial (same convention the hardcoded times use).
- Strikes outside the PC recording's time coverage can't be edited there;
  they are carried over from the previous source and reported on save.
- The file is re-loaded by the picker — and honored by `rerun_gait.sh` —
  only when its first line contains `pick_heel_strikes.py`; the old
  hand-written `heel_strikes.txt` files stay inert.

Workflow: pick → `bash rerun_gait.sh` (regenerates logs, PNGs and the
per-trial CSVs with the new strikes) → Stage 3 plotters.

---

## Stage 2: Batch runner — `rerun_gait.sh`

36 commands (18 trials × PC/SD), one per run. Each command reproduces the
settings of the existing `*_imu_analysis.log` exactly: the per-trial EMG
channel, `--env-cutoff 15`, the per-session gain (1 or 8), the heel
strikes (picked `heel_strikes.txt` for that trial if present, otherwise
the log-derived hardcoded times scaled by 1000/980 for the 980 sps time
base), `--emg-fs 980`,
default percent time axis, `--no-plot --save-results
--save-trial-csv`.

Rerunning it regenerates all logs, PNGs and the per-trial CSVs with the
current script version:

```bash
bash rerun_gait.sh
```

---

## Stage 3: Plotting — `plot_gait_trials.py`

```
.venv/bin/python plot_gait_trials.py            # all trials, interactive
.venv/bin/python plot_gait_trials.py --no-plot  # full sweep: save only
.venv/bin/python plot_gait_trials.py --source PC   # PC recordings only
.venv/bin/python plot_gait_trials.py --walk-type Incline
.venv/bin/python plot_gait_trials.py --csv path/to/BLE_8_Max_Fast_Walk_2026-09-14_PC_signals.csv
.venv/bin/python plot_gait_trials.py --csv ... --raw-emg --channel 3
```

Walks `application_testing/` for `*_signals.csv` + `*_emg.csv` pairs inside
any `*Walk*` trial folder and reads **only the CSVs** (no log parsing).

**Interactive popups**: every figure is written to `--out-dir` *and* left
open, then `plt.show()` blocks at the end so each figure appears as a
window with matplotlib's pan/zoom/scroll toolbar (use `--no-plot` for
batch runs — a full sweep would otherwise open one window per figure;
headless runs without a `DISPLAY` fall back to saving only).

**`--csv`** restricts the run to a single trial: pass the path of its
`*_signals.csv` (the sibling `*_emg.csv` is derived; the raw capture CSV
`<stem>.csv` or a bare filename searched under `--base` are also accepted).

**`--raw-emg [--channel N]`** replaces the EMG-envelope subplot with the
**raw, unfiltered** EMG of channel `chN` (default 1) — no bandpass, no
notch, no rectification, no envelope — read straight from the capture CSV
named by the `# input:` header comment (codes → µV → mV, DC offset kept,
gap rows drawn as line breaks). The raw overlay uses a 2048-point cycle
grid (instead of 201) so the waveform survives the resampling, and one
extra figure per trial, `<trial_id>_raw_ch<N>.png`, shows the channel as a
plain time series with the heel strikes marked (no cycle locking, no
mean/CI).

### Method

1. **Cycle extraction**: the heel strikes from the `_signals.csv` header
   are paired into gait cycles (consecutive strikes: strike k starts cycle
   k, strike k+1 ends it and starts the next). Each signal is interpolated
   onto a 0–100 % cycle grid (201 points); cycles with fewer than 10 valid
   samples are skipped.
2. **Signals plotted** (one subplot each): EMG envelope (µV), dynamic
   accel PC1 (g), angular-rate PC1 (deg/s) — the same three signals as the
   per-trial `*_gait_cycles.png`.
3. **Per-trial figure** (`<trial_id>_cycle_overlays.png`): individual cycle
   traces (grey dashed) + mean (black) + 90 % CI, like the sit-stand
   cycle overlays.
4. **Cross-trial figure** (`gait_cycle_means.png`): every trial's mean
   curve on one subplot per signal, one colour per trial.
5. **`gait_plot_data.csv`**: exactly the plotted numbers — one row per
   trial × signal × cycle-% point: `trial_id, signal, cycle_pct, mean,
   ci_low, ci_high, n_cycles`.

### Decisions

- **CSVs, not logs**: the full signal traces are needed here, so the
  analysis writes them directly; the header comments replace log parsing.
- **0–100 % normalization** (default): gait-cycle durations vary between
  trials and subjects, so percent normalization makes the cross-trial mean
  curves comparable. (`--time-axis time` support in the plotter follows the
  per-trial script if ever needed.)
- **90 % CI** = 1.645 · SD / √n, same convention as the per-trial overlays.

---

## Stage 3b: Group plotter — `plot_gait_group.py`

```
.venv/bin/python plot_gait_group.py
```

Reads the same per-trial CSVs as Stage 3 and produces **one group figure
per walk type** — `gait_incline_cycle_overlays.png`,
`gait_fast_cycle_overlays.png`, `gait_slow_cycle_overlays.png` — comparing
the four **gain-1 PC** recordings, one trial per quadrant:

| Quadrant | Trial |
|----------|-------|
| top-left | `BLE_1_Karl_<Walk>_Walk_2026-09-08_PC` |
| top-right | `BLE_1_Max_<Walk>_Walk_2026-09-14_PC` |
| bottom-left | `USB_1_Karl_<Walk>_Walk_2026-09-02_PC` |
| bottom-right | `USB_1_Max_<Walk>_Walk_2026-09-02_PC` |

Each quadrant has **4 stacked subplots** of gait-cycle-locked overlays
(0–100 % cycle, individual traces + mean + 90 % CI, same style as Stage 3):

1. **xyz Euler angles** — roll / pitch / yaw from `_signals.csv`
   (`roll_deg`, `pitch_deg`, `yaw_deg`), one colour per angle
2. EMG envelope (µV)
3. dynamic accel PC1 (g)
4. angular-rate PC1 (deg/s)

The gain filter reads the `# gain:` header comment of the `_emg.csv`
(the trial-id rep digit is checked as well), so `--gain 8` would pick the
BLE_8 trials instead. `--source SD` switches to the SD variants;
`--walk-type` restricts to one walk type.

Also writes `gait_plot_data.csv` with exactly the plotted numbers — one
row per trial × signal × cycle-% point: `trial_id, signal, cycle_pct, mean,
ci_low, ci_high, n_cycles` (same schema as the Stage 3 CSV, plus the three
Euler-angle signals).

### Decisions

- **Reuses Stage 3 helpers** (`find_trials`, `load_trial`, `cycle_curves`,
  the `imu_t0_s` shift, the 90 % CI convention) by importing them from
  `plot_gait_trials.py`, so both plotters stay consistent.
- **One figure per walk type, not one combined figure**: three 2×2
  quadrant figures keep the four subject × connection combinations side by
  side per walk condition without overloading a single figure.
- **Euler angles on top**: the top subplot shows the cycle-locked
  orientation (roll/pitch/yaw) because it has no PCA sign ambiguity,
  unlike the PC1 signals.

---

## Stage 3c: Walk-type overlay figure — `plot_gait_group_overlay.py`

```
.venv/bin/python plot_gait_group_overlay.py                 # RMS boxes (default)
.venv/bin/python plot_gait_group_overlay.py --box-metric p2p # peak-to-peak boxes
```

One figure — `gait_group_overlay.png` — comparing the three walk types
over USB and BLE, **Max only** (Karl is not considered), gain-1 PC
recordings. Layout is a **3 × 4 matrix**:

| | col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|---|
| row 1 | USB overlays — EMG envelope (µV) | USB boxes (metric) | BLE overlays — EMG envelope | BLE boxes (metric) |
| row 2 | USB — dyn accel PC1 (g) | USB boxes | BLE — dyn accel PC1 | BLE boxes |
| row 3 | USB — ang rate PC1 (deg/s) | USB boxes | BLE — ang rate PC1 | BLE boxes |

- **Overlay cells**: Slow / Fast / Incline each drawn as mean + 90 % CI —
  Slow `tab:blue`, Fast `tab:green`, Incline `tab:red`.
- **Box-plot cells** (directly right of their overlay): one coloured box
  per walk type showing the **per-cycle metric** distribution (~10
  values per box), selected with `--box-metric`:
  - `rms` (**default**) — `sqrt(mean(y^2))` over each cycle's samples,
    as recorded (DC offset included); ylabel `RMS (unit)`.
  - `p2p` — each cycle's max − min; ylabel `peak-to-peak (unit)`.
- **Style** (sit-stand style): cycle panels are fully open — **all
  spines removed** (bottom spine only on the bottom row), no y tick
  notches (labels kept), one uniform y-text size, dashed gray grid —
  and share **one x axis** whose tick marks and labels appear **only on
  the bottom row**. The x axis title `% of gait cycle (0 = heel strike)`
  is set **twice** — directly under each bottom-row ang-rate overlay,
  never centred under the figure. The figure's **only legend** sits in
  the top-left subplot. Box plots keep their framed, categorical
  (walk-type) x axis.

Also writes `gait_overlay_plot_data.csv` (`connection, walk_type,
trial_id, signal, cycle_pct, mean, ci_low, ci_high, n_cycles`) and
`gait_overlay_box_data.csv` (`connection, walk_type, trial_id, signal,
cycle, metric, value` — BOTH box metrics, `metric` ∈ {`rms`, `p2p`},
regardless of the `--box-metric` used for the figure).

---

## Reproducing the whole pipeline

```bash
cd /home/karl/repos/Sokosti_tools

# 1. per-trial analyses (regenerates logs + PNGs + per-trial CSVs)
bash imu_gait_tools/rerun_gait.sh

# 2. plots (figures + gait_plot_data.csv into imu_gait_tools/; --no-plot
#    = save only, otherwise one interactive window per figure opens)
.venv/bin/python imu_gait_tools/plot_gait_trials.py --no-plot

# 2b. group figures (gain-1 PC trials, 2x2 quadrants per walk type)
.venv/bin/python imu_gait_tools/plot_gait_group.py

# 2c. walk-type overlay figure (Max only, 3x4 matrix with box plots)
.venv/bin/python imu_gait_tools/plot_gait_group_overlay.py
```

Python environment: the workspace venv (`.venv/bin/python`, Python 3.12) has
numpy/scipy/pandas/matplotlib installed.

---

## Known caveats

- **BLE gap rates vary widely** between trials; the EMG envelope is computed
  gap-aware per contiguous valid segment, and the gap statistics are in each
  log's EMG sample-loss table.
- **Manual heel strikes** are subjective; the originals are recorded in
  each log's `# Command:` line and in each `_signals.csv` header
  (`heel_strikes_s`), so the exact cycles can always be audited. With
  Stage 1b the authoritative source per trial is a picked
  `heel_strikes.txt` (which itself records its source CSV in a header
  comment) once `rerun_gait.sh` has been re-run with it.
- **PCA sign is arbitrary**; PC1 is canonicalised by convention (angular-
  rate PC1: axis pitch component positive, "pitch-up = positive"; dynamic-
  accel PC1: largest-|component| axis entry positive), so PC1 polarity is
  consistent across trials, connections (BLE/USB) and sources (PC/SD) and
  the BLE/USB and cross-trial mean curves are directly comparable.
- **Different EMG channels per subject/session** (Karl ch16, Max-USB ch2,
  Max-BLE ch3): the envelope amplitudes are not directly comparable across
  subjects; compare patterns, not absolute µV.
- **SD variants** are analyzed and plotted by default; pass `--source PC` to
  restrict the plots to the PC recordings (as in the Sit_Stand pipeline).