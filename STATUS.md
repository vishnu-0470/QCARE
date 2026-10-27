# Latest follow-up status — 2026-10-26

- **PASS:** >90% quantized-model target: 90.2673% DS2 accuracy, macro-F1 0.360308.
- **PASS:** ten fusion RTL vectors and four digital board selections.
- **PASS:** generic fusion-board synthesis; no physical timing/power claim.
- **PASS:** stronger software experiment evaluated: 93.3425%, macro-F1 0.410956.
- **FAIL:** >98% accuracy target.
- **NOT TESTED/BLOCKED:** physical FPGA/ADC/AFE, Vivado timing, original MATLAB stage tests, and Spectre.
- **PASS:** MATLAB frozen-export replay for ten reference vectors (the older QNN path).

Use [RUN_GUIDE.md](RUN_GUIDE.md) and [latest report](reports/LATEST_REPORT.md).
All following baseline results remain preserved and refer to the earlier model.

---

# Q-CARE status — 2026-10-26

**Accuracy and latency targets were not achieved.** Software, digital RTL and generic synthesis results are recorded below. Physical FPGA and AFE acceptance remain open.

| Check | Status | Result and evidence |
|---|---|---|
| Original preservation | **PASS** | Original ZIP, 180-file hash inventory and pre-change backups included. [backups/original_project.zip](backups/original_project.zip) |
| Dataset | **PASS** | All 132 publisher-verified files for 44 records parsed; record 119 repaired. [reports/dataset_manifest.json](reports/dataset_manifest.json) |
| Preparation | **PASS** | 100,679 complete-window beats; 54 boundary exclusions; real ECG training only. [reports/dataset_preparation.json](reports/dataset_preparation.json) |
| Training | **PASS** | 300 epochs: 30 floating and 270 QAT; 18 DS1 fit and 4 DS1 validation records. [reports/training_execution.json](reports/training_execution.json) |
| QAT memorization diagnostic | **PASS** | Ten real DS1 beats memorized; diagnostic accuracy 100%. [reports/real_qat_memorization.json](reports/real_qat_memorization.json) |
| DS2 evaluation | **PASS** | 49,688 beats from all 22 DS2 records; RR and QNN score identical beat identities. [reports/ds2_metrics.json](reports/ds2_metrics.json) |
| Accuracy >99.6% | **FAIL** | Observed 69.7553%, macro-F1 0.258583. [reports/ds2_metrics.json](reports/ds2_metrics.json) |
| Original golden replay | **PASS** | 8/8 exact; all-F predictions reproduce the original initialization. [reports/baseline/qnn.log](reports/baseline/qnn.log) |
| Trained RTL replay | **PASS** | 10/10 real DS2 vectors: all five signed logits and classes match. [reports/trained_qnn_checked.log](reports/trained_qnn_checked.log) |
| Batched float32 QAT parity | **FAIL** | Numerical deviations observed; frozen inference uses float64 integer emulation. [reports/batched_inference_diagnostic.json](reports/batched_inference_diagnostic.json) |
| Frozen inference oracle | **PASS** | All 21 operation tensors and batched final logits match on ten real DS2 beats. [experiments/ds2_evaluation/EVALUATED.json](experiments/ds2_evaluation/EVALUATED.json) |
| Annotated-beat trace | **PASS** | Record 100, annotation 2, R sample 370 traced through CWT, memory and class N. [reports/beat_trace.json](reports/beat_trace.json) |
| Legacy frozen streaming replay | **PASS** | 267/267 frozen events; a separate detector/tree/small-CNN path. [reports/baseline/streaming_frozen_replay.json](reports/baseline/streaming_frozen_replay.json) |
| MATLAB frozen-export replay | **PASS** | User ran the Windows replay; ten classes/logits and exported memory order matched. This covers the older QNN path. [matlab/sim/verify_current_export.m](matlab/sim/verify_current_export.m) |
| Cadence CSV replay | **PASS** | All 1,501 supplied ADC codes and 3,600 legacy map tokens reproduced. [reports/cadence_adapter.json](reports/cadence_adapter.json) |
| Trained Cadence RTL replay | **PASS** | 1/1 exact signed-logit replay; no classification ground truth for this waveform. [reports/afe_trained_checked.log](reports/afe_trained_checked.log) |
| Board wrapper simulation | **PASS** | 4/4 selections: logits, LEDs and display segments; scan reset repaired. [reports/trained_board_checked.json](reports/trained_board_checked.json) |
| Board constraints | **PASS** | All 21 used pin assignments checked against Digilent master constraints. [xilinx/rtl/basys3/basys3.xdc](xilinx/rtl/basys3/basys3.xdc) |
| Generic xc7 synthesis | **PASS** | Core and board mapped netlists produced; zero static-check problems. [reports/synthesis/trained_board/result.json](reports/synthesis/trained_board/result.json) |
| Latency conversion | **ESTIMATED** | 883,718 simulated cycles imply 8.83718 ms at an assumed achieved 100 MHz. [reports/performance_summary.json](reports/performance_summary.json) |
| Latency <0.35 ms | **FAIL** | 8.83718 ms under the declared 100 MHz assumption; host CWT and acquisition excluded. [reports/PERFORMANCE.md](reports/PERFORMANCE.md) |
| Vivado implementation/timing | **BLOCKED** | Vivado unavailable; no placement, routing or timing closure result. [xilinx/fpga/vivado_qcare.tcl](xilinx/fpga/vivado_qcare.tcl) |
| Power/energy <0.07 mJ | **NOT TESTED** | No measured power or energy; no fabricated energy estimate. [reports/PERFORMANCE.md](reports/PERFORMANCE.md) |
| Programming/physical ADC | **NOT TESTED** | No FPGA programming or physical ADC acquisition executed. [reports/trained_board_checked.json](reports/trained_board_checked.json) |
| AFE electrical acceptance | **BLOCKED** | No Spectre runtime or schematic/VM database in the confirmed ZIP; earlier _ocn_ path is absent. [xilinx/sim/AFE_CADENCE_COSIM.md](xilinx/sim/AFE_CADENCE_COSIM.md) |
| Post-synthesis equivalence | **NOT TESTED** | No formal or gate-level functional equivalence run. [reports/PERFORMANCE.md](reports/PERFORMANCE.md) |
| Provenance guards | **PASS** | 132 raw files, four caches and checkpoint provenance checked; changed/missing inputs rejected. [reports/provenance_checks.json](reports/provenance_checks.json) |
| Other research experiments | **NOT TESTED** | No intra-patient retraining, TTA, replacement fusion model or AFE-conditioned DS2 run. [README.md](README.md) |

PASS applies only to the named check. It does not imply accuracy-target, timing, clinical or physical acceptance. The five-class QNN remains separate from the legacy streaming detector. CWT runs on the host and the board demo uses stored maps. Conventional records 201/202 share a subject across DS1/DS2.

See [README](README.md), [final report](reports/FINAL_REPORT.md), [engineering audit](reports/ENGINEERING_AUDIT.md), and [performance](reports/PERFORMANCE.md).
