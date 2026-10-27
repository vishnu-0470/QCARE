# Run the project

## Run the existing Verilog checks in Vivado on Windows

The RTL and testbenches are already Verilog (`.v`) files. Icarus is only
required by the existing Python wrappers; this Vivado XSim flow runs the same
checked benches without Icarus.

1. Open Vivado and show the Tcl Console (**Window → Tcl Console** if it is
   hidden).
2. Run these commands in that console:

   ```tcl
   cd C:/QCARE_FPGA
   source xilinx/fpga/vivado_xsim_fusion90.tcl
   ```

The script runs ten fusion-engine vectors and four stored-map board-wrapper
selections. Each invocation writes compile, elaboration, and run logs under a
new timestamped directory in `reports/vivado_xsim/`. This is digital
simulation; it does not program a physical board. The script uses fresh output
folders and preserves the existing exports and reports.

Run commands from the project root. The commands below use the already
prepared project artifacts; they do not retrain or re-evaluate the frozen
models. In Windows PowerShell, open the project folder and run:

```powershell
Set-Location C:\QCARE_FPGA
python python\show_latest_results.py
python python\infer_record_fusion90.py --record 100 --beat-index 0
python python\infer_record_software.py --record 100 --beat-index 0
```

The fusion demo reads a real WFDB ECG beat, computes its CWT, loads the frozen
model, and writes the prediction and input map. It does not retrain anything.
On macOS/Linux, `./run.sh` provides equivalent shortcuts when the documented
`.venv` is installed. `RUN_ALL.command` is a macOS launcher.

In VS Code, select **Terminal → Run Task**, then choose **Q-CARE: Show saved
metrics**, **Q-CARE: Fusion inference demo**, or **Q-CARE: Software inference
demo** to see each command's output in the integrated terminal. Python source
stays grouped in `python/`; `.vscode/tasks.json` stores these editor run tasks.

## Run the MATLAB replay

Open MATLAB with `C:\QCARE_FPGA` as the current folder, then run:

```matlab
addpath(genpath(fullfile(pwd,'matlab')));
run_qcare_set('verify')
```

The replay was run successfully on this Windows workstation: exact logits,
classes, and exported ROM ordering matched for all ten frozen vectors. It
verifies the older `ds1_current` QNN export, not the newer fusion network.

To run all four staged checks from the MATLAB Command Window, use:

```matlab
run_qcare_set('stages')
```

It prints each stage's results and saves inspection/export files under
`experiments/`. To process the supplied waveform CSV, use
`run_qcare_set('waveform')`. The workflow sets are described in
[matlab/README.md](matlab/README.md).

## Reported follow-up results

The measured follow-up results are **90.2673% accuracy / 0.360308 macro-F1** for
the 4W/8A fusion network, and **93.3425% / 0.410956** for the separate software
classifier. **The >98% target was not achieved.** Both use all 22 DS2 records and
49,688 beats; previous DS2 results were known, so these are follow-up results.
The software classifier has not been exported as a 4-bit FPGA network.

## What the commands do

| Command | What runs | Outputs |
|---|---|---|
| `./run.sh results` | Prints saved measured metrics | Terminal |
| `./run.sh demo --record 100 --beat-index 0` | Full CWT + causal features + quantized fusion replay | `experiments/replay/record100_beat0/` |
| `./run.sh demo-software --record 100 --beat-index 0` | Extended-feature software inference | `experiments/software_replay/record100_beat0/` |
| `./run.sh rtl` | Icarus self-checking fusion engine, ten real vectors | `reports/accuracy90/RTL_fusion.log` |
| `./run.sh board-sim` | Four switch selections, logits, LEDs and display checks | `reports/fusion_board_checked.log` |
| `./run.sh verify` | Publisher/source/cache/checkpoint hash checks | `reports/provenance_checks.json` |
| `./run.sh plots` | Rebuilds plots from saved results | `reports/plots/accuracy_followup/` |
| `./run.sh synth` | Generic Xilinx board synthesis | `reports/synthesis/fusion_board/` |

For another real beat, change `--record` and `--beat-index`. The index counts
complete 360-sample annotated windows within that record, starting at zero.
Every replay uses preceding unlabeled beats for causal template/RR state. It
never fits on the annotation class being predicted. The supplied data contain
the conventional 44 records, not all 48 records in the full MIT-BIH database.

A single-beat RTL replay can use the demo's generated input:

```sh
.venv/bin/python python/run_fusion90_checks.py \
  --export experiments/replay/record100_beat0 \
  --vectors 1 --log reports/accuracy90/record100_RTL.log
```

The fusion input has 3616 signed bytes per beat: 3600 row-major CWT codes followed
by 16 auxiliary feature codes. The loader writes the auxiliary values to RAM
addresses 32064..32079. Keep the fusion program, weights, biases, normalization
and vector files together. Old 3600-token CWT-only vectors cannot be paired with
the fusion model.

## Requirements and setup on a different machine

Use Python 3.12. Do not move a copied virtual environment between operating
systems. Create it again and install the executed versions:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

When the environment is installed, use `./run.sh` or `.venv/bin/python`
directly on macOS/Linux; activation is unnecessary. Do not copy a virtual
environment between operating systems.

RTL checks require `iverilog` and `vvp` on PATH. They are already installed on
this host. On a new Mac with Homebrew, install them using:

```sh
brew install icarus-verilog
```

Generic synthesis used YoWASP Yosys 0.48 with the supplied runtime. The local
`tools/yosys_stable` package is used by `./run.sh synth` when present. For a fresh
environment, install `xilinx/fpga/requirements-yosys.txt`. These are generic mapped
counts, not Vivado timing or physical measurements.

## Data preparation and model reproduction

The project folder includes the 44 verified real records and full prepared cache.
The ZIP excludes large regenerable cache arrays. To regenerate a cache in a
fresh working copy:

```sh
.venv/bin/python python/download_dataset.py
.venv/bin/python python/prepare_dataset.py
.venv/bin/python python/check_provenance.py
```

The downloader rewrites the before/after audit manifest. Use a separate copy if
you want to preserve the original incomplete-record audit. The transform writes
`experiments/cache/maps.npy`, `augmented_maps.npy`, `beats.npz` and preparation
metadata. No synthetic ECG training is used.

The delivered models are frozen and deliberately refuse accidental retraining
or repeated full-DS2 evaluation. To reproduce the final neural follow-up, first
make a separate project copy or preserve the experiment directory under a new
name. The existing DS1 feature caches and DS1-only teacher are already supplied:

```sh
mv experiments/accuracy90_fusion_mlp experiments/accuracy90_fusion_mlp_saved
.venv/bin/python python/train_fusion_mlp90.py
.venv/bin/python python/evaluate_fusion90.py
./run.sh rtl
./run.sh board-sim
```

This trains only the new quantized fusion head; the original CWT/IRN backbone
stays frozen. Training uses real DS1 features, 80 floating head epochs and 180 QAT
head epochs. Candidate selection uses the four DS1 validation records. The
DS1-only HGBT teacher contributes soft targets on those same real fit beats;
it is not the deployed FPGA classifier. Rerunning after seeing DS2 is an
exploratory follow-up, not a new pristine independent test.

The additional source scripts show every executed trial:

1. `calibrate_accuracy90.py`: FC-bias prior adjustment, DS1 only.
2. `refit_accuracy90.py`: refit original FC5x16, DS1 only.
3. `refit_fusion90.py`: linear GAP/rhythm fusion, DS1 only.
4. `diagnose_fusion90.py`: DS1-only software diagnostics/teacher.
5. `train_fusion_mlp90.py`: the final nonlinear 4W/8A fusion head.
6. `evaluate_fusion90.py`: frozen fusion evaluation and integer export.
7. `train_accuracy98.py`: two stronger extended-feature HGBT candidates.
8. `train_accuracy98_forest.py`: ExtraTrees/RandomForest comparison.
9. `evaluate_accuracy98.py`: DS1-selected software follow-up evaluation.

These scripts have fixed evidence directories and frozen guards. Preserve or
rename those directories in a research copy before reproducing them. The
baseline's 300-epoch trainer is `train_current.py`; its measured old model and
reports remain intact and should not be confused with the improved model.

## FPGA, MATLAB and Cadence

- Fusion engine: `xilinx/rtl/qnn/qnn_engine.v`, configured with NOPS=23, NW=10248, NB=181.
- Fusion board wrapper: `xilinx/rtl/basys3/basys3_qnn_fusion_top.v`.
- Checked benches: `xilinx/sim/tb_qnn_fusion_checked.v`, `tb_board_fusion_checked.v`.
- Frozen export: `experiments/accuracy90_fusion_mlp/export/`.
- User-run Vivado flow: `vivado -mode batch -source xilinx/fpga/vivado_fusion90.tcl`.

Vivado/MATLAB/Spectre were unavailable during authoring. The supplied Vivado
flow is NOT TESTED and does not program a board. Generic synthesis and digital
simulation do not establish timing closure, power, ADC connection or physical
operation. CWT and auxiliary feature extraction remain on the host.

The retained MATLAB sources and `matlab/sim/verify_current_export.m` replay the old
CWT-only architecture; they do not constitute an executed MATLAB port of the
new nonlinear fusion model. Python is the tested current inference path.
Cadence instructions in `xilinx/sim/AFE_CADENCE_COSIM.md` refer to the separately
preserved CWT-only co-simulation. Do not feed those old vectors to the new
fusion model. AFE gain/noise/bandwidth/overload/CMFB/bandgap checks remain open.

## Where to find outputs

- Latest measured comparison: `reports/LATEST_REPORT.md` and `LATEST_RESULTS.json`.
- Fusion full-DS2 predictions: `experiments/accuracy90_fusion_mlp/DS2_followup/`.
- Software full-DS2 predictions: `experiments/accuracy98_selected/`.
- Confusion pictures: `reports/plots/accuracy_followup/`.
- Original and Cadence CWT maps: `experiments/beat_trace/`, `experiments/afe_adc_trained/`.
- Per-beat new CWT picture/data: demo's `cwt_map.csv`, `beat.csv`, input memory and JSON.
- Recoverable originals: `backups/original_project.zip` and per-file backups.
- Complete current change inventory: `reports/modified_files.json`.

The >98% result is not supported by these experiments. Rare S/F/Q performance
remains weak and is explicitly reported. The dataset/protocol and class order
N,S,V,F,Q were not changed to manufacture a higher accuracy.
