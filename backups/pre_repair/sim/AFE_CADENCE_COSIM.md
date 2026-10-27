# Cadence AFE waveform into the ECG FPGA classifier

## What this adapter does

It reads the existing `2.csv` differential output in the project root, models a
12-bit offset-binary ADC over -1 V to +1 V at 500 samples/s, and reconstructs
the quantized waveform. It resamples that waveform to 360 samples/s because the
project's CWT front end expects 360 samples, finds an automatic peak estimate,
then creates the 60x60 signed 8-bit CWT map used by the current QNN hardware.

The automatic peak estimate is the largest absolute deviation from the signal
median. Cadence has no R-wave annotation in this CSV, so review the selected
peak time printed by MATLAB. This is an inference smoke input, not evidence of
clinical accuracy or of a physical ADC connection.

## Run in MATLAB

From the project root, run:

```matlab
run('sim/prepare_afe_qnn_input.m')
```

This writes `rtl/qnn/vec/afe_qnn_vec_in.mem` in the row-major byte format the
existing QNN engine consumes. It does not overwrite the project's original
`qnn_vec_in.mem` golden vectors.

## Run the RTL inference in the Linux VM

From the project root, run:

```sh
cd rtl/qnn
iverilog -g2012 -o ../../sim/afe_qnn.vvp qnn_engine.v ../../sim/tb_afe_qnn.v
vvp ../../sim/afe_qnn.vvp
```

The bench prints the predicted class and five logits. It uses the project's
existing generated network weights and program memory.
