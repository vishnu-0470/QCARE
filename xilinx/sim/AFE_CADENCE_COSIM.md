# Cadence waveform / ADC / QNN boundary

**PASS — software waveform replay and digital co-simulation executed.** The checked-in `data/2.csv` is treated as a differential voltage export. No Spectre job, schematic electrical check, physical ADC connection or FPGA measurement was executed. There is no annotation or ground-truth class for this waveform, so its predicted class is not an accuracy result.

## Declared behavioral interface

| Boundary | Definition |
|---|---|
| Source | `data/2.csv`; 796 finite data rows, 3.0 s; one duplicated timestamp averaged |
| Differential ADC range | −1 V to +1 V, provisional |
| ADC encoding | 12-bit offset binary, endpoint-inclusive behavioral model |
| ADC sample clock | Uniform 500 samples/s, 1,501 samples for this source |
| Quantization | `clip(floor((v+1)/2*4095+0.5),0,4095)` |
| Reconstruction | `-1 + code*2/4095` volts; modeled step = 488.400488 µV |
| Resampling | Linear interpolation from reconstructed 500 Hz samples to 360 Hz |
| Peak selection | Maximum absolute deviation from the waveform median among indices supporting a complete beat |
| Selected peak | Zero-based index 746 at 2.072222222 s; MATLAB one-based index 747 |
| Beat window | 140 samples before and 219 after peak, total 360 |
| CWT | Current Stage 1 cosine-Morlet transform; 60 scales × 60 temporal bins; per-map max-absolute normalization |
| QNN input | `clip(floor(128*M+0.5),-128,127)`; signed int8, 3,600 row-major bytes |

The ADC range, resolution, clock and reference/common-mode compatibility require the intended ADC datasheet and circuit interface. The CSV alone cannot establish them. This endpoint-inclusive quantizer is a declared software model, not an asserted physical converter transfer function. No off-grid final sample is appended. Linear interpolation preserves the original adapter method; no additional antialias filtering claim is made.

The peak heuristic uses the waveform median and is noncausal. MIT-BIH benchmark beats instead use official annotation R locations. The benchmark was not conditioned through this Cadence path, and its accuracy must remain separate from this one-waveform replay.

## What was verified

- **PASS:** all 1,501 supplied codes in `data/adc_test_vector_2_12bit_500sps.csv` match reconstruction from `data/2.csv`; observed codes are 449…3456, with zero rail codes.
- **PASS:** the legacy `round(127*M)` path reproduces every byte of the original `xilinx/rtl/qnn/vec/afe_qnn_vec_in.mem`. That frozen input remains unchanged.
- **PASS:** the corrected canonical 128-scale input is generated separately under `experiments/afe_adc_checked/`. This fixes the mismatch between the former standalone helper and the model's `e_in=-7` contract.
- **PASS:** the corrected input with original weights matches all five Python integer logits in RTL (`reports/afe_qnn_checked.log`).
- **PASS:** the same waveform with the trained frozen export matches all five logits in RTL (`reports/afe_trained_checked.log`): `[23,-118,-9,-170,131]`, class Q (index 4). This class has no supplied ground truth.
- **FAIL, original bench:** its 200,000-cycle timeout was shorter than the approximately 883,718-cycle engine execution. The replacement uses a two-million-cycle fatal timeout and checks logits and class, rather than merely printing a prediction.
- **BLOCKED:** MATLAB/Octave replay and new Spectre/OCEAN simulation; these runtimes were unavailable. The supplied project has no Cadence schematic database/VMware project to simulate. The existing waveform and any original circuit assets are preserved.

The source SHA-256 is `2cdcdbcc519618cd13f9227584d27152637fe6f2fd0f357f254664870d8cf7e0`. Detailed reports and intermediate ADC voltages, codes, segment and CWT map are saved in the corresponding experiment directories.

## Replay with Python and Icarus

From the project root, using the project's Python environment:

```sh
python python/prepare_cadence.py --export xilinx/rtl/qnn --out experiments/afe_adc_checked
python python/run_rtl_checks.py --export experiments/afe_adc_checked --model-export xilinx/rtl/qnn --vectors 1 --log reports/afe_qnn_checked.log
python python/prepare_cadence.py --export experiments/ds1_current/export --out experiments/afe_adc_trained
python python/run_rtl_checks.py --export experiments/afe_adc_trained --model-export experiments/ds1_current/export --vectors 1 --log reports/afe_trained_checked.log
```

The model export must contain `gen/qnn_prog.mem`, its packed companion `gen/qnn_prog.mem.wide`, `gen/qnn_w.mem` and `gen/qnn_b.mem`. The adapter output contains `vec/qnn_vec_in.mem` and `vec/qnn_vec_exp.txt`. Each expected row is `class logitN logitS logitV logitF logitQ`, using signed decimal logits. The input file contains one two's-complement hex byte per line, scale row first, temporal column fastest.

The general runner supplies absolute paths and places all Icarus parameter options before source filenames. The legacy convenience wrapper can also be compiled from the project root:

```sh
iverilog -g2012 -s tb_afe_qnn -o reports/afe_wrapper.vvp xilinx/rtl/qnn/qnn_engine.v xilinx/sim/tb_qnn_checked.v xilinx/sim/tb_afe_qnn.v
vvp reports/afe_wrapper.vvp
```

This convenience wrapper uses the corrected `experiments/afe_adc_checked` vectors with original `xilinx/rtl/qnn/gen` weights. Use the Python runner for the trained export.

## MATLAB replay source

The source-level MATLAB bridge is retained in `matlab/sim/prepare_afe_qnn_input.m`. Run from the project root:

```matlab
afe_output_root = fullfile(pwd,'experiments','afe_adc_matlab');
run('matlab/sim/prepare_afe_qnn_input.m');
```

The output directory is explicit so Python and MATLAB intermediates remain reviewable side by side. The script saves `matlab_adapter_values.mat` and the row-major input memory. Do not pair this input with an unrelated expected-logit file. The adapter now loads `experiments/ds1_current/frozen_model.mat` by default and writes expected logits using its matching `Qm`; set `afe_model_file` to explicitly select another saved model. The optional model must match the RTL ROMs. If no model is found, only the input map is written. The checked Python adapter/runner above supplies both input and expected outputs. MATLAB execution of this bridge is **BLOCKED on the current host**, not marked as passed from inspection.

## Outstanding physical work

The waveform replay does not verify 38 dB closed-loop gain, 0.5–150 Hz bandwidth, 4.5 µVrms input-referred noise, rail clamping, 180 ms overload recovery, CMFB stability or bandgap behavior. Those require the actual circuit simulation outputs and test conditions. Representative annotated beats may later be driven through that preserved circuit for a separate robustness experiment. Full-record conditioning would require a declared, consistent protocol before reporting conditioned DS1/DS2 accuracy.
