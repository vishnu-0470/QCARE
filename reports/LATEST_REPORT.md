# Latest accuracy experiments — 2026-10-26

**The >98% target was not achieved.** Both new models were selected using DS1 validation only. The original models, original ZIP and earlier results remain preserved.

| Model | DS2 accuracy | Macro-F1 | Deployment status |
|---|---:|---:|---|
| Quantized CWT/rhythm fusion | 90.2673% | 0.360308 | 4W/8A/B16; integer export and RTL checked |
| Extended-feature software classifier | 93.3425% | 0.410956 | Software only; no quantized RTL claim |

Both score the same 49,688 annotated beats from all 22 conventional DS2 records. Eighteen DS1 records fit each model and four DS1 records validate/select it. No random beat split or DS2 label fitting was substituted. Records 201/202 share one individual in the conventional split. Because DS2 results were already known from earlier experiments, these are explicitly follow-up exploratory results, not a fresh untouched test-set claim.

The quantized model retains the original 14-convolution backbone and adds a 32-input/32-hidden nonlinear fusion layer before FC5. It combines GAP16 with 16 signed 8-bit rhythm/morphology proxies. Global feature normalization uses DS1-fit records only; patient-relative template/RR state uses preceding unlabeled beats and resets for every record. No future RR or patient identity is used as an input. Width/polarity/template measures are heuristic proxies rather than clinically validated PQRST delineation. CWT and auxiliary feature extraction remain on the host.

The software model uses 208 features: original GAP16, morphology16, causal RR16, normalized real waveform80, and causal waveform-template residual80. Two HGBT and two forest configurations were compared on DS1 validation; the selected HGBT was frozen before its DS2 follow-up. It cannot be presented as FPGA accuracy.

| Model | Class | Support | Sensitivity | Precision | F1 |
|---|---|---:|---:|---:|---:|
| Quantized fusion | N | 44236 | 0.948164 | 0.946645 | 0.947404 |
| Quantized fusion | S | 1837 | 0.044638 | 0.532468 | 0.082371 |
| Quantized fusion | V | 3220 | 0.877019 | 0.682620 | 0.767704 |
| Quantized fusion | F | 388 | 0.007732 | 0.002752 | 0.004060 |
| Quantized fusion | Q | 7 | 0.000000 | 0.000000 | 0.000000 |
| Software | N | 44236 | 0.978185 | 0.953022 | 0.965440 |
| Software | S | 1837 | 0.071856 | 0.571429 | 0.127660 |
| Software | V | 3220 | 0.913043 | 0.755590 | 0.826888 |
| Software | F | 388 | 0.095361 | 0.229814 | 0.134791 |
| Software | Q | 7 | 0.000000 | 0.000000 | 0.000000 |

Rare-class performance remains weak, especially S/F/Q. The higher overall accuracy is not evidence of uniform five-class reliability or clinical readiness. Full confusion matrices, specificity, supports, bootstrap intervals and saved per-beat predictions are included in each experiment.

The fusion export has 23 operations, 10,248 signed 4-bit weights and 181 signed 16-bit biases. All ten real DS2 RTL vectors match every signed 24-bit logit. Four stored-map board selections and a separate record100 replay pass. Generic synthesis reports 2,615 LUT primitives, 745 FFs, 21 RAMB36, 7 RAMB18 and 24 DSPs. Measured simulation count is 884,953 engine cycles; 8.84953 ms at an assumed 100 MHz is ESTIMATED and does not meet 0.35 ms. Timing closure, power, programming and physical operation remain NOT TESTED.

Run `./run.sh results`, `./run.sh demo`, `./run.sh demo-software`, `./run.sh rtl`, or `./run.sh board-sim` from the project folder. See `RUN_GUIDE.md` for installation, complete reproduction commands, output locations, optional MATLAB/Cadence and synthesis steps. The supplied local `.venv` is for this Mac; recreate it from requirements on another machine.
