# Personalised Federated Learning for Wearable Activity Recognition

Activity recognition (18-class classification) from smartphone and smartwatch
accelerometer and gyroscope data (WISDM dataset). The project builds centralised
baselines, simulates federated averaging across participants, and studies how
fine-tuning on a small labelled support set helps an unseen participant.

This is a single-machine simulation. It does not provide formal privacy
guarantees, clinical claims, or a multi-device deployment.

## Research questions

- How do centralised training and federated averaging compare on the same
  held-out participants and evaluation windows?
- Does supervised fine-tuning on a small support set improve a global model on
  separate query windows from an unseen participant?
- Minor: heterogeneity across participants, and phone vs watch inputs (and
  fusion, only if cross-device time alignment can be verified).

## Dataset

[WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
(51 participants, 18 activities, phone and watch accelerometer and gyroscope,
nominally 20 Hz). The data is not included in this repository. Download and
extract it, then point `RAW_DIRECTORY` in the notebook at
`wisdm-dataset/raw/`. See `WISDM-dataset-description.pdf` for the official
description.

## Repository layout

| Path | Description |
|---|---|
| `NCP-WISDM.ipynb` | Main notebook: parsing, data audit, segmentation, windowing |
| `Data-Audit.odt` | Week 1 data audit write-up |
| `Project-Scope.odt` | Project scope document |
| `Research/` | Reference papers |
| `parse_errors.csv` | Raw lines rejected by the parser, with file, line and reason (header only if none) |
| `conflict_log.csv` | Duplicate timestamps with conflicting values inside a segment (header only if none) |
| `Label_Conflict_Log.csv` | Rows sharing a file and timestamp but labelled with different activities |
| `exclusions.csv` | Every excluded row, segment or unpaired segment, with a level and reason |
| `exclusions_by_user_class.csv` | Accepted/rejected windows and excluded rows per user and activity (chosen feature mode) |
| `window_manifest.csv` | Every candidate watch window of the chosen mode: validity, status, rejection reason, raw-source lines |
| `window_residual.csv` | Accel/gyro start/end residuals for every paired window |
| `user_eligibility.csv` | Usable windows and missing classes per user, checked before the split is frozen |
| `split_manifest.csv` | Frozen participant split (user, split, seed) |
| `frozen_rules.json` | Frozen window, validity, pairing and split settings |
| `alignment_audit.md` | Accel/gyro and phone/watch clock evidence, tolerances, rejection counts and decisions |
| `cross_device_audit.csv` | Phone vs watch span offsets per (subject, activity) |
| `features_watch.parquet`, `feature_columns.csv` | Watch feature table with trace columns, and the explicit feature-column list |

## Pipeline (Weeks 1-2)

1. **Parse** all 204 raw files into one table, validating every line (6 fields,
   integer subject/timestamp, finite x/y/z) and logging rejects to
   `parse_errors.csv`; cache it as `wisdm_raw.parquet` with each row's raw
   `Source_Line`.
2. **Audit**: coverage, missing files/activities, duplicates, clock resets,
   sampling rates, block lengths.
3. **Segment** each file at activity changes, clock resets (any backwards
   step) and long gaps (forward step over 0.5 s) before anything else.
   Separate occurrences of an activity are never concatenated.
4. **Clean** without sorting: resolve duplicate timestamps (keep first, log
   conflicts), remove rows whose timestamp is shared by two activities, drop
   exact-copy (replayed) segments, then assert time strictly increases inside
   every segment. Timestamps stay as integer nanoseconds; a local origin is
   subtracted before conversion to seconds.
5. **Pair accel/gyro segments by time overlap** (one-to-one, largest overlap
   first) inside each subject, device and activity, verify the clock
   relationship, and set a common origin per pair.
6. **Window** (10 s non-overlapping windows) and compute per-sensor validity
   statistics directly on retained samples (no interpolation). Rules:
   at least 160 samples per sensor, observed span of at least 8 s, no gap
   over 0.5 s. A separate accel-only fallback windows every watch accel
   segment on its own origin.
7. **Check eligibility, then split** by participant (60/20/20 = 31 train /
   10 eval / 10 test, seed 42). Rules are judged on training subjects only,
   then frozen.
8. **Pair windows** (start/end residuals within 0.25 s), choose the feature
   mode (watch accel+gyro if at least 95% of rule-valid training windows
   match, otherwise the 15-feature accel fallback) and extract features.
9. **Audit the phone/watch clock**. Fusion is only allowed on shared-clock
   evidence and is off for Week 2.

## Data findings so far

- Subject 1629: every activity is recorded twice as an exact copy (72 segments
  dropped).
- Subject 1649 (watch): 1,160 accel and 1,159 gyro samples for activities M and O
  share identical timestamps and values. These rows are excluded from both
  activities (4,638 rows in total).
- Accel/gyro clocks overlap in time for 1,819 of 1,820 candidate pairs. The
  remaining pair (subject 1642, phone, activity J) is a mis-pairing.
- The phone gyroscope starts later than the accelerometer by a median of
  about 0.15 s (watch: 0.05 s), so a 0.25 s start tolerance rejects many phone
  pairs. This is to be audited in the pairing step.
- Sampling rates are not uniform (20 Hz for most files; some 25 Hz and 50 Hz).

## Status

- [x] Data audit
- [x] Segmentation, duplicate resolution, common origin, windowing
- [x] Accel/gyro window pairing and residual audit (or 15-feature watch-accel fallback)
- [x] Saved user-disjoint splits and cross-device clock audit
- [ ] Logistic regression and MLP baselines
- [ ] Federated averaging (10 rounds, 5 clients per round, 1 local epoch)
- [ ] Personalisation experiments

## Running

```bash
pip install pandas numpy polars scikit-learn matplotlib pyarrow
pip install torch --index-url https://download.pytorch.org/whl/cpu
jupyter notebook NCP-WISDM.ipynb
```

The full table is about 0.75 GB per copy in memory (15.6M rows). The cleaning
cells make several copies, so 8 GB of RAM is tight; restart the kernel and
reload from the parquet cache if memory is low. The validating parser peaks at
about 2.2 GB and takes about a minute, so only re-run it when the cache needs
rebuilding.
