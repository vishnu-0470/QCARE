# Latest follow-up results and quick start

The best measured software accuracy is **93.3425%** (macro-F1 **0.410956**).
The improved **4W/8A fusion network** scores **90.2673%** (macro-F1 **0.360308**)
and passes ten signed-logit RTL vectors plus four digital board selections.
**The requested >98% target was not achieved.** These are DS1-selected follow-up
experiments on the same conventional DS2 records whose earlier results were
already reported. The software score is not an FPGA-network score.

Start with **[RUN_GUIDE.md](RUN_GUIDE.md)** for Windows PowerShell and
macOS/Linux commands.
Current measured details: [LATEST_REPORT](reports/LATEST_REPORT.md).
The earlier audit below remains preserved as historical baseline documentation.

---

# Q-CARE FPGA — audited continuation

This project contains the Q-CARE ECG classification sources, frozen model artifacts, MIT-BIH input records, RTL simulations, and engineering reports. `backups/original_project.zip` preserves the original supplied project archive; the other backup folders preserve selected pre-repair and pre-release files.

**The requested >99.6% accuracy was not achieved.** The frozen current quantized network obtained **69.7553% accuracy and 0.258583 macro-F1** on 49,688 annotated beats from all 22 conventional DS2 records. These are beat-classification results. Physical FPGA operation and AFE electrical performance are not established.

| Held-out DS2 method | Accuracy | Macro-F1 | Evaluation status |
|---|---:|---:|---|
| Frozen current IRN, quantized | 69.7553% | 0.258583 | PASS — evaluation executed |
| Same frozen latent parameters, quantization disabled | 67.6059% | 0.249232 | PASS — evaluation executed |
| DS1-selected floating warm-start checkpoint | 72.9130% | 0.284425 | PASS — evaluation executed |
| RR logistic baseline | 63.4781% | 0.298131 | PASS — evaluation executed |
| Requested >99.6% accuracy | — | — | FAIL |

The RR and neural methods use the same fit/validation/test records, annotations, boundary exclusions and test beat identities. The RR baseline has higher macro-F1 despite lower overall accuracy. See `reports/ds2_metrics.json`, `experiments/ds2_evaluation/predictions.csv`, the confusion matrices in `reports/plots/`, and `reports/ENGINEERING_AUDIT.md` for per-class results and limitations. `STATUS.md` is the acceptance-check index.

## What is connected

The current five-class path is:

```text
MIT-BIH annotation + ECG → 360-sample beat → host CWT → 60×60 signed-int8 map
    → qnn_engine → five signed logits → N / S / V / F / Q
```

`python/qcare/model.py` ports the existing MATLAB architecture: downsampling convolution, IRN1, IRN2, max pool, downsampling convolution, IRN3, IRN4, global average pool, FC5. It contains **14 convolutions, four IRN blocks, 9,144 weights and 149 biases**. This continuation does not replace it with the paper's six-layer architecture. CWT is computed on the host. The QNN RTL is a microcoded layer-at-a-time executor, not a complete sample-to-class streaming CWT implementation.

`rtl/ecg_afe_accel_top.v` is a separate legacy detector/quality-gate/feature/tree/small-CNN path with a two-bit class interface. Its frozen regression is retained, but it is not the five-class CWT/IRN model. `train_all.m` does not train the current QNN. `rtl/basys3/basys3_qnn_top.v` demonstrates four stored maps; it has no physical ADC input connection.

## Data and frozen experiment

All required `.hea`, `.dat` and `.atr` files for **44/44 records** were verified against the PhysioNet publisher SHA-256 manifest and parsed with WFDB. Record 119's incomplete data and missing metadata/annotations were repaired; previous files are recoverable. The four paced records 102, 104, 107 and 217 are outside this conventional protocol.

- Class order is `N,S,V,F,Q`, with indices `0,1,2,3,4` throughout software, golden files, RTL and LEDs.
- Select MLII where present, otherwise channel zero. Use 360 Hz physical ECG samples, annotation R-peak index, 140 samples before and 219 after it; reject incomplete base windows.
- CWT: scales 1–60, cosine Morlet with `omega=pi/2`, support ±4 scale units, translation stride 2, average pool 3, per-beat max-absolute normalization. The fused transform yields one 60×60 map.
- Input quantization: `clip(floor(128*M+0.5), -128, 127)`. Memory is scale-row-major, temporal column fastest, two's-complement bytes.
- DS1 fit uses 18 records; validation uses DS1 records **108, 114, 209, 223**. DS2 is excluded from fitting, scaling calibration, checkpoint selection and early stopping.
- The conventional assignment places **201 in DS1 and 202 in DS2, although both are from the same person**. Describe this as the conventional DS1/DS2 record split, not as strictly subject-disjoint.
- Fit/validation/test contain 41,743 / 9,248 / 49,688 beats, respectively. The corresponding boundary exclusions are 26 / 4 / 24. No required record was silently omitted.
- Training uses only real annotated ECG. Rare-class augmentation shifts actual windows by −20…+20 samples (`b-60`, `b=40…80`); shifted windows may be zero-padded only at record boundaries. No new synthetic ECG was used for training.

The run uses seed 26, momentum SGD (0.8), initial learning rate 0.01, batch size 200, training-only FC-input dropout 0.5, and cross-entropy. It runs 30 floating warm-up epochs plus 270 QAT epochs with a cosine learning-rate schedule per phase. Each epoch selects up to 5,000 N and V beats and 2,000 S/F/Q samples each, with replacement where needed. These choices and deviations from the paper are recorded in `experiments/ds1_current/config.json`.

The best quantized checkpoint was selected by DS1 validation macro-F1 at zero-based epoch 41, then frozen before DS2 evaluation. Its SHA-256 is:

```text
311e9af2537cbad16e9539c80cb2f2a8804d80dd5421f141c2502af563705be5
```

`FROZEN.json` and `experiments/ds2_evaluation/EVALUATED.json` record that boundary. A failed numerical export check preceded the completed evaluation; its log is retained. Batched float32 QAT calculations differed from the independent integer oracle. Frozen quantized evaluation therefore uses float64 arithmetic on the same power-of-two grids, with all 21 operation tensors checked on ten real DS2 beats. **Weights, scales and checkpoint selection were not changed after observing DS2.** Float64 here is software emulation; hardware remains integer. The original float32 training/validation history remains intact and this numerical limitation is documented.

## Reproduce checks on the delivered artifacts

Run commands from the project root. A Python 3.12 virtual environment with NumPy, SciPy, PyTorch, WFDB, scikit-learn, joblib, threadpoolctl and Matplotlib is required; install the pinned packages with `python -m pip install -r requirements.txt`; exact executed versions are also recorded under `reports/`. Icarus Verilog (`iverilog` and `vvp`) must be on `PATH` for RTL checks. No MATLAB runtime is required for the Python path.

```sh
python python/check_baseline.py
python python/run_rtl_checks.py --export experiments/ds1_current/export --model-export experiments/ds1_current/export --vectors 10 --log reports/trained_qnn_checked.log
python python/run_board_checks.py
python python/prepare_cadence.py --export experiments/ds1_current/export --out experiments/afe_adc_trained
python python/run_rtl_checks.py --export experiments/afe_adc_trained --model-export experiments/ds1_current/export --vectors 1 --log reports/afe_trained_checked.log
python python/trace_frozen_beat.py
python python/plot_results.py
```

These commands replay the frozen artifacts; they do not train a replacement model. The traced beat is record 100, annotation index 2, R sample 370 (zero-based), annotated N, exported logits `[177,-46,-102,-96,26]`, predicted N. The generated trace includes its raw beat, CWT map, integer operation outputs, addresses and hashes.

The trained export passed **10/10 RTL vector checks**, comparing all five signed logits and the selected class. These ten beats comprise two examples per annotated class, not the full DS2 set. The board wrapper passed **4/4 stored-map checks**, including full logits, class LEDs, beat-index display and class display. This is digital simulation only.

## Regenerate data and rerun training

Use a separate copy of the delivered project for a new experiment, and preserve its existing `experiments/ds1_current`, `experiments/ds2_evaluation`, `experiments/rr_baseline` and reports before running commands that write those paths. The scripts use fixed paths for this audited experiment; do not overwrite the delivered evidence casually. The large, regenerable CWT/augmentation cache is not required to replay the exported RTL vectors and may be excluded from the ZIP.

```sh
python python/download_dataset.py
python python/prepare_dataset.py
python python/train_current.py --run experiments/ds1_current --epochs 300 --float-epochs 30 --seed 26 --threads 4 --stop-after 30
python python/train_current.py --run experiments/ds1_current --epochs 300 --float-epochs 30 --seed 26 --threads 1 --resume experiments/ds1_current/last.pt
python python/fit_rr_baseline.py
python python/evaluate_frozen.py
python python/plot_results.py
```

The two training commands reproduce the executed four-thread floating phase and one-thread QAT phase. The actual initial transition failed on safe loading of PyTorch `TorchVersion` metadata; the repaired loader narrowly permits that metadata class, and the run resumed from its saved state. Logs retain both phases and the failure. Floating arithmetic can vary with platform and library backend; the delivered frozen weights and golden memories are the authoritative inference artifacts.

The evaluator requires a valid frozen-checkpoint hash and refuses to run if the completed evaluation marker already exists. Preserve the completed evaluation; do not remove the guard to tune against DS2. No TTA, focal-loss model, rhythm-feature fusion, intra-patient retraining or quality-gate-based DS2 filtering was substituted into this baseline.

## Quantization, RTL and physical limits

Weights are signed four-bit values stored in 16-bit ROM words. Biases remain signed **16-bit accumulator-scale** values, matching the existing RTL. Inputs are signed eight-bit; post-ReLU intermediate activations are restricted to `[0,127]`. Accumulation is signed 32-bit, final logits are signed 24-bit, and ties select the first class. Shift requantization rounds with `floor(x+0.5)` before saturation.

The repaired instruction ROM uses a companion `qnn_prog.mem.wide`: each 288-bit line packs the same eighteen 16-bit fields, with opcode in the least-significant field. Keep this file with `qnn_prog.mem`, `qnn_w.mem` and `qnn_b.mem`. Both Python and MATLAB exporters generate it. Original golden files and original weights remain under `rtl/qnn/`; trained files are under `experiments/ds1_current/export/`.

The checked testbench reports **883,718 engine cycles**. At a hypothetical achieved 100 MHz this is **8.83718 ms** engine latency, exceeding the 0.35 ms target; this clock-to-time conversion is ESTIMATED, not a measured post-route or board result. It excludes host CWT, input acquisition and input loading. Power and energy targets remain NOT TESTED. Consult `STATUS.md` and synthesis reports for the final synthesis outcome; partial tool output alone is not a complete resource or timing result.

MATLAB/Octave, Vivado and Cadence/Spectre were unavailable on this host. Original MATLAB stage tests, post-route timing, bitstream generation, FPGA programming, physical ADC acquisition and AFE electrical acceptance remain BLOCKED or NOT TESTED as individually recorded. The supplied ZIP contained waveform exports but no Cadence schematic database or VMware project to rerun.

## Sources and attribution

- [PhysioNet MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/) and its [publisher checksums](https://physionet.org/files/mitdb/1.0.0/SHA256SUMS.txt). Data attribution and licensing must accompany redistribution; PhysioNet lists the Open Data Commons Attribution License v1.0.
- [MIT-BIH introduction and recording notes](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm), including the 201/202 shared-subject caveat.
- [Published conventional DS1/DS2 record assignments](https://pmc.ncbi.nlm.nih.gov/articles/PMC3142238/) and [de Chazal et al., 2004](https://doi.org/10.1109/TBME.2004.827359).
- [Digilent Basys 3 master constraints](https://github.com/Digilent/digilent-xdc/blob/master/Basys-3-Master.xdc): the 21 used pin assignments were checked against this source. This does not establish board timing or operation.
- The user-supplied Cao et al. PDF is the architecture/settings reference. Its text supports the listed SGD starting settings but does not disclose the conventional DS1/DS2 lists. This continuation is not an exact replication or a demonstrated improvement over its 99.5% result.


## Release integrity and synthesis replay

`python python/check_provenance.py` verifies raw data, the regenerated cache and
checkpoint preparation metadata. The cache arrays are omitted from the ZIP;
regenerate them before this check. Training and evaluation reject changed
inputs and accidental writes over frozen evidence. These guards were added
after the recorded run and checked without retraining or reevaluating DS2.

Completed generic synthesis used the isolated release pinned in
`fpga/requirements-yosys.txt`. Install it in a separate virtual environment,
then run `python python/synthesize.py --yosys <path-to-yowasp-yosys>` and repeat
with `--board`. The resulting structured reports and internal `-l` logs are
required; a successful process exit alone is insufficient. Actual mapped
wrapper counts are 2,592 LUT primitives, 737 FFs, 20 RAMB36, 7 RAMB18 and 24 DSPs.
`python python/performance_summary.py` regenerates the summary from saved
synthesis and simulation evidence. The Vivado Tcl flow in
`fpga/vivado_qcare.tcl` is provided for the target `xc7a35tcpg236-1` and was
NOT TESTED here. It does not program a board.

For MATLAB, use `addpath('sim'); verify_current_export` to compare the frozen
`.mat` state, ten real golden vectors and reexported ROM ordering. This replay
is BLOCKED on the authoring host because MATLAB/Octave is unavailable.
