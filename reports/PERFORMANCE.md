# Executed performance and feasibility

**Accuracy target: FAIL. Latency target: FAIL under the declared 100 MHz assumption. Energy result: NOT TESTED.**

| Scope | LUT primitives | FF | RAMB36E1 | RAMB18E1 | DSP48E1 | Status |
|---|---:|---:|---:|---:|---:|---|
| QNN core | 2543 | 647 | 20 | 0 | 23 | PASS — generic synthesis |
| Stored-map Basys 3 wrapper | 2592 | 737 | 20 | 7 | 24 | PASS — generic synthesis |

Counts come from YoWASP Yosys 0.48 `synth_xilinx -family xc7`, with frozen trained ROMs. The wrapper totals include its QNN submodule. Both flows completed `check` with zero reported problems. Reports, mapped EDIF/JSON and exact ROM/source hashes are in `reports/synthesis/`. This is generic 7-series synthesis, not Vivado utilization, placement, routing, timing closure or physical verification.

The ten-vector RTL check measured 883,718 engine clock cycles per vector. At an **assumed achieved 100 MHz**, the derived latency is **8.83718 ms (ESTIMATED)**, versus the requested <0.35 ms. The same budget allows only 35,000 cycles. Clock frequency has not been demonstrated after routing. Host CWT, map loading, signal acquisition and button debounce are excluded. The beat itself requires 219 post-R samples (0.6083 seconds at 360 Hz), a separate algorithmic look-ahead.

Power and energy have not been measured. No LUT-based energy guess is substituted. If an explicit power assumption is supplied, `--assumed-power-w` produces an ESTIMATED value using `E_mJ = P_W × latency_ms`. Such a value is not a measured acceptance result.

The supplied paper Table I reports 11,364 LUTs, 21,118 FFs, 0 DSPs, 28 BRAMs, 0.350 ms, FPGA-core 0.219 W/0.0767 mJ, and whole-board 1.02 W/0.357 mJ for its balanced design. These are paper-reported values. Its CWT is in FPGA; ours is on the host. Its 25.4 Kbit model and six-layer architecture differ from this preserved 9,293-parameter model. Its exact DS1/DS2 protocol is not disclosed in the supplied paper. These differences prevent a like-for-like superiority claim.

The original narrow program ROM was accessed at eighteen independent addresses per instruction. The repaired engine reads one 288-bit instruction assembled offline from the same eighteen 16-bit fields. The original word files and arithmetic remain compatible; the `.mem.wide` companion is required. Original eight-vector parity and new trained ten-vector parity pass. No formal or post-synthesis netlist equivalence run was performed.

Earlier tool failures and report-parser failures are retained under `reports/baseline/`, `reports/synthesis/report_path_failure/` and `reports/synthesis/board_report_parser_failure.json`. A console stream stopped printing at ABC, so final internal `-l` logs, structured statistics and output netlists are used as completion evidence.

Vivado and a physical Basys 3 were unavailable. Timing/implementation is BLOCKED; programming, physical ADC acquisition, board power and energy remain NOT TESTED. A user-run Vivado Tcl flow is provided and labeled unexecuted.
