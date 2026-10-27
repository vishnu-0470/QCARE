# Q-CARE executed results — 2026-10-26

**The requested >99.6% accuracy was not achieved.** The frozen current quantized model obtained **69.7553% accuracy and 0.258583 macro-F1** on 49,688 annotated beats from all 22 DS2 records. The report documents a real-data baseline and repaired verification path; it does not establish an improvement over the paper.

| Saved method | Accuracy | Macro-F1 | 95% record-bootstrap accuracy interval |
|---|---:|---:|---|
| quantized_current_irn | 69.7553% | 0.258583 | 57.08%–81.66% |
| float_same_latent_parameters | 67.6059% | 0.249232 | 54.21%–79.81% |
| float_warm_start_checkpoint | 72.9130% | 0.284425 | 62.76%–82.38% |
| RR_logistic_baseline | 63.4781% | 0.298131 | 52.43%–75.36% |

Every method uses the same fit, validation and test records, annotation mapping, boundary exclusions and scored test beat identities. Fitting uses 18 DS1 records; validation uses 108, 114, 209 and 223; testing uses all 22 DS2 records. Included beats are 41,743 / 9,248 / 49,688, with incomplete-window exclusions of 26 / 4 / 24. No required record is missing. Records 201 and 202 are the same person in the conventional split, so strict subject independence cannot be claimed.

Uncertainty intervals use 1,000 record-cluster bootstrap resamples with seed 26. They describe variation across the fixed DS2 records. Q has only seven test beats and eight DS1 beats; validation contains no Q. Macro-F1 averages all five classes, including zero-valued F1. These intervals do not establish independent-patient clinical validation.

| Model | Class | Support | Sensitivity | Positive predictivity | F1 | Specificity |
|---|---|---:|---:|---:|---:|---:|
| Quantized IRN | N | 44236 | 0.717583 | 0.945520 | 0.815932 | 0.664527 |
| Quantized IRN | S | 1837 | 0.111595 | 0.068152 | 0.084623 | 0.941422 |
| Quantized IRN | V | 3220 | 0.842236 | 0.255753 | 0.392361 | 0.830163 |
| Quantized IRN | F | 388 | 0.000000 | 0.000000 | 0.000000 | 0.992779 |
| Quantized IRN | Q | 7 | 0.000000 | 0.000000 | 0.000000 | 0.956764 |
| RR logistic | N | 44236 | 0.662221 | 0.988994 | 0.793273 | 0.940205 |
| RR logistic | S | 1837 | 0.122482 | 0.102975 | 0.111885 | 0.959040 |
| RR logistic | V | 3220 | 0.566770 | 0.465324 | 0.511061 | 0.954872 |
| RR logistic | F | 388 | 0.507732 | 0.040163 | 0.074438 | 0.904503 |
| RR logistic | Q | 7 | 0.000000 | 0.000000 | 0.000000 | 0.817717 |

The QNN has no true-positive F or Q beats, and S sensitivity is 11.16%. Both trained methods have accuracy below the N-class prevalence of 89.0275% in DS2. The RR baseline has higher macro-F1 than the QNN. Accuracy alone therefore obscures the class-imbalance problem. RR features use past annotated intervals and ratios, a fit-only StandardScaler and class-balanced logistic regression (C=1); an independent R detector was not benchmarked.

Confusion matrices use true classes as rows and predictions as columns, ordered N, S, V, F, Q. All metrics and raw matrices are in `reports/ds2_metrics.json`; saved per-beat predictions are in `experiments/ds2_evaluation/`. Neither model used DS2 for fitting, calibration, early stopping or checkpoint selection. A mechanical inference-parity repair followed a failed initial export check: the frozen weights and scales stayed unchanged, while quantized evaluation used float64 power-of-two integer emulation. The initial failure and numerical diagnostic remain available. Float32 training and validation selection are preserved as executed.

The current architecture remains unchanged: 14 convolutions, four IRN blocks, 9,144 weights and 149 biases. Its formats are signed 4-bit weights, 8-bit inputs, post-ReLU activations in [0,127], signed 16-bit biases, 32-bit accumulators and 24-bit logits. It differs from the paper architecture. The 30 floating plus 270 QAT epochs, frozen-scale calibration and cosine schedule are declared choices. Training used real annotated ECG and shifted real windows only.

Executed consistency checks include the original 8/8 QNN vectors, 10/10 new real DS2 vectors (every signed logit), all 21 operation tensors on ten real beats, 4/4 board-wrapper selections including display segments, one trained Cadence-waveform RTL replay, and 267/267 legacy frozen streaming events. These vector suites do not mean that all 49,688 DS2 beats were simulated in RTL. The original all-F initialization remains preserved.

Generic Xilinx 7-series synthesis passed. The core uses 2,543 LUT primitives, 647 FFs, 20 RAMB36 blocks and 23 DSPs. The full stored-map wrapper uses 2,592 LUT primitives, 737 FFs, 20 RAMB36 plus 7 RAMB18 blocks and 24 DSPs. Both completed static checks with zero reported problems. These are mapped synthesis counts, without placement or timing proof.

RTL simulation counts 883,718 engine cycles. At an assumed achieved 100 MHz, derived engine latency is 8.83718 ms (ESTIMATED), above the <0.35 ms target. Host CWT, map loading, acquisition and debounce are excluded. Power, energy and physical operation remain NOT TESTED. Vivado implementation, MATLAB reruns and Cadence/Spectre runs are BLOCKED by unavailable tools. Corresponding source and instructions are included.

The provisional ADC is 12-bit offset-binary over ±1 V at 500 Hz, followed by linear interpolation to 360 Hz. Cadence peak selection is heuristic, with no annotation or class ground truth. CSV replay does not verify AFE gain, noise, bandwidth, clamping, overload recovery, CMFB or bandgap behavior.

The refreshed ZIP includes all 44 verified source records, the original archive and backups, code, RTL, constraints, trained checkpoints, reports and separate golden vectors. Large regenerable CWT arrays and compiled simulator executables are excluded. Preparation metadata and regeneration commands are included. See `README.md`, `STATUS.md`, `reports/ENGINEERING_AUDIT.md`, `reports/PERFORMANCE.md`, `reports/modified_files.json` and `data/ATTRIBUTION.md`.
