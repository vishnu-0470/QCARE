# MATLAB code

Open MATLAB with `C:\QCARE_FPGA` as the current folder, then add the MATLAB
source tree to the path:

```matlab
addpath(genpath(fullfile(pwd,'matlab')));
```

Start with the replay that compares the saved MATLAB integer model against the
ten frozen export vectors:

```matlab
verify_current_export
```

It prints a `PASS` message when every class, logit, and exported memory image
matches. The replay writes its comparison export under
`experiments/ds1_current/matlab_reexport/` and leaves the saved model untouched.

## Folders

- `cwt/` contains the continuous-wavelet-transform front end and map scaling.
- `irn/` contains the floating-point IRN neural-network layers and gradients.
- `qat/` contains quantization, integer inference, and export routines.
- `sim/` contains runnable MATLAB replay and stage scripts. The Xilinx
  Verilog benches are kept separately under `xilinx/sim/`.

Useful MATLAB entry points in `sim/`:

- `verify_current_export.m` checks the frozen MATLAB/Python export.
- `prepare_afe_qnn_input.m` converts the CSV waveform into an ECG map and
  writes reviewable adapter outputs under `experiments/afe_adc_matlab/`.
- `t_stage1.m` through `t_stage3.m` inspect the CWT, network, and quantization
  stages. These are development experiments and can take longer to run.
- `t_stage4_export.m` writes its generated model to
  `experiments/matlab_stage4_export/`.

See the root `RUN_GUIDE.md` for the Vivado and Python run steps.
