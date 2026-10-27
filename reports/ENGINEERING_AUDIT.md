# Engineering audit — 2026-10-26

This continuation uses the updated ZIP confirmed current by the user. The live
Windows path was inaccessible on this macOS host. The original ZIP and its
180-file SHA256 inventory are included in `backups/`, with individual pre-change
copies. No Cadence schematic or VMware database appeared in the supplied ZIP.
The earlier `_ocn_` attachment path no longer exists. No circuit was recreated.

## Baseline and integration boundaries

The existing five-class architecture is the MATLAB Stage 1 CWT followed by the
Stage 2/3 Inception-ResNeXt and `rtl/qnn/qnn_engine.v`. It has four IRN blocks and
14 convolutions, with 9,293 parameters. The board wrapper consumes four stored
CWT maps and has no physical ADC input. The CWT remains a host computation.

`rtl/ecg_afe_accel_top.v` is a separate legacy streaming detector, quality gate,
feature/tree/small-CNN path with a two-bit class interface. Its original
`train_all.m` is incompatible with the current five-class model. A frozen rE
replay matched 267/267 events, but regenerated per-case regressions require
Octave. The old regression script swallowed missing-tool errors; it now fails
closed. Parameter options precede Verilog input filenames for Icarus portability.

The original eight QNN vectors matched bit-exactly and all predicted F. Four
of those original verification maps were random; they were preserved and were
not used for new training. The archived model was an initialization export,
not an established trained classifier. The original Cadence testbench timed out
at 200,000 cycles, below the actual 883,718-cycle checked execution.

## Data and training

All 44 conventional records and 132 source files were retrieved or reused from
publisher-verified official files. Record 119 had truncated data and missing
header/annotations; originals were backed up and repaired. Record 100's modified
header was restored to the publisher version. Both manifest snapshots remain.

Class order is N/S/V/F/Q. The existing annotation map is retained. The 360-sample
window uses 140 samples before and 219 after each annotation. MLII is selected
where available, otherwise channel zero. Incomplete base windows are excluded;
shifted real training views may be zero-padded at record boundaries. Exactly
54 mapped beats are excluded, leaving 100,679 beats. No synthetic ECG training
was performed. Required record lists and per-record counts are explicit in the
manifest; none are silently omitted.

Eighteen DS1 records fit the model; four DS1 records (108,114,209,223) validate it.
All 22 DS2 records are isolated from fitting, calibration and model selection.
The conventional split retains 201/202 across DS1/DS2 despite their shared
subject; it is not strictly subject-disjoint. The supplied Cao paper does not
publish these lists, so its 99.5% cannot be treated as a matched inter-patient
baseline. Sources are recorded in `reports/reference_sources.json` and README.

The PyTorch port preserves the current architecture and bias precision. Training
uses SGD, initial LR .01, momentum .8, batch 200 and FC-input dropout .5. The
chosen 30-epoch floating warm-up, 270 QAT epochs, cosine schedule, balanced
sampling and frozen calibration are declared deviations, not exact replication.
A safe-load metadata error at the transition was repaired narrowly and the
saved state/RNG resumed; four CPU threads were used for floating epochs and one
for QAT after a real-batch benchmark. Configuration, source identity, versions,
commands and selected epochs are in `reports/training_execution.json`.

A ten-real-beat diagnostic reached 100% memorization, showing that the repaired
warm-started QAT path can learn these examples. The old random-input MATLAB
memorization test was not rerun. Generalization still fails: held-out accuracy
is 69.7553%, macro-F1 .258583, with zero F/Q true positives. The same-split RR
baseline gives 63.4781% and .298131. Both methods score the same 49,688 beats.
All five classes, supports, specificity, precision/recall/F1, record-bootstrap
intervals and raw predictions are saved. The target >99.6% is FAIL.

## Numerical and export repairs

Canonical input quantization is `clip(floor(128*M+.5),-128,127)`, consistent with
QNN input exponent -7. The original helper's `round(127*M)` differed and is
preserved in backups; the old Cadence vector can still be reproduced separately.
The saved MATLAB CWT text reference matches Python to 4.99961e-6, within its
six-significant-digit tolerance of 6e-6. This is not a current MATLAB execution.

New frozen inference uncovered batch-dependent float32 deviations. Float64
power-of-two emulation matches the independent integer oracle on all 21
operation tensors and batched final logits for ten real DS2 beats. The frozen
checkpoint, scales and DS1-only checkpoint choice were not changed. The initial
failed check remains in `reports/ds2_evaluation_initial_failure.log`. Original
float32 training/validation remains as executed; its numerical behavior is a
limitation. The completed evaluation is guarded against accidental reruns.

Logical 4-bit weights use 16-bit ROM containers; biases remain signed B16;
post-ReLU activations are [0,127]; accumulation is signed32 and logits signed24.
Rounding is floor(x+.5), saturation explicit, and argmax ties choose the first
class. The new testbench rejects missing tokens, malformed expected rows,
timeouts and every signed-logit mismatch. Ten trained vectors pass in RTL.
The labeled beat trace also validates MATLAB struct/ROM ordering in Python.
MATLAB execution of `sim/verify_current_export.m` remains BLOCKED.

The instruction ROM now stores one 288-bit word per operation, preserving the
same eighteen 16-bit fields and arithmetic. Both exporters emit the companion
`.mem.wide` file. The old program, weights and vectors remain preserved. Basys
seven-segment scanning lacked reset and produced unknown simulation output;
reset was added and all four switch selections, logits, LEDs and display
segments now pass the checked wrapper testbench.

Post-run provenance guards stream-check all source/cache hashes, enforce the
44-record split and compare checkpoint preparation metadata. Current files
pass; isolated changed/missing-file fixtures are rejected. No retraining or
DS2 inference was performed to test these guards.

## Cadence and FPGA feasibility

The supplied `2.csv` has one duplicate timestamp, averaged before interpolation.
The provisional ADC is 12-bit offset-binary, endpoint-inclusive ±1 V at 500 Hz;
all 1,501 supplied ADC codes replay exactly. The adapter linearly resamples to
360 Hz, selects heuristic peak index 746, takes the same 360-sample window and
writes a row-major 60x60 map. The original 3,600-token map replays exactly under
the legacy quantizer. Corrected/trained maps are separate, and trained RTL
replay passes one vector. This waveform has no class ground truth. Gain,
noise, bandwidth, rail-clamping, overload, CMFB, bandgap and physical ADC
assumptions remain unverified; no Spectre run was performed.

The 21 used Basys 3 pin assignments match Digilent's master XDC. Generic Yosys
0.48 Xilinx mapping completed for core and wrapper, with zero static-check
problems. The wrapper uses 2,592 LUT primitives, 737 FFs, 20 RAMB36, 7 RAMB18
and 24 DSPs. `reports/PERFORMANCE.md` explains tool failures, corrected reporting,
source/ROM hashes and the mapped netlists. Counts are not Vivado placement
results. At an assumed achieved 100 MHz, 883,718 simulated engine cycles imply
8.83718 ms, exceeding .35 ms. No physical timing, power or energy claim is made.
Vivado is unavailable; the supplied Tcl flow is unexecuted. Formal/post-synthesis
functional equivalence, bitstream generation, programming and physical board
operation remain NOT TESTED or BLOCKED as indexed in `STATUS.md`.
