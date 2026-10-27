# Release review

**PASS — artifact and saved-metric checks executed.** See
`release_validation.json`, `provenance_checks.json` and `guard_rejections.json`.
No new training or DS2 inference was run during release validation.

Review found no DS2 records in the actual fit, validation or activation-scale
calibration index sets. The conventional 201/202 shared-subject caveat remains.
Saved metrics were independently recomputed from saved predictions with
scikit-learn and match all four reports. The frozen checkpoint hash is unchanged.

Two reproducibility weaknesses were corrected after the experiment: preparation
previously trusted a stale manifest, and training/evaluation did not enforce
recorded cache hashes. Current guards verify source and cache hashes before use
and compare checkpoint preparation metadata. Existing frozen runs and the
completed DS2 evaluation reject accidental overwrites. These changes do not
alter model architecture, data, learned values, selection or saved predictions.

The RR arithmetic metadata was corrected from a generic float32 label to
float64 sklearn processing. A misleading comment about grouping 201/202 was
removed: the reported bootstrap clusters DS2 records, where only 202 occurs.
Neither correction changes predictions or intervals.

Known material limitations remain: failed accuracy/latency targets, no F/Q
true positives, poor S recall, very small Q support, no Q validation beats,
batch-dependent float32 QAT numerical behavior, a changed architecture from the
paper, host-side CWT, separate legacy streaming classifier, no post-synthesis
functional equivalence, and unavailable MATLAB/Vivado/Spectre/physical hardware.
These are disclosed in STATUS, the engineering audit and performance report.
