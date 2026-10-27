function c = hw_cfg()
% HW_CFG  Single source of truth for every constant shared by the MATLAB
% golden model and the RTL (exported to xilinx/rtl/gen/params.vh by export_rtl.m).
% All sample indices in the golden model are 0-based (like the RTL counters).

c.fs = 360;                       % MIT-BIH sampling rate (Hz)

% ---- streaming QRS detector (integer Pan-Tompkins-style) --------------
c.det.mwi_sh  = 5;                % moving window = 2^5 = 32 samples (89 ms)
c.det.tmin    = 8;                % absolute floor of the threshold (ADC codes)
c.det.track   = 24;               % samples tracked after a threshold crossing
c.det.refr    = 72;               % refractory period (200 ms) from crossing
c.det.npk_sh  = 8;                % noise-peak estimator rate  (1/256)
c.det.spk_sh  = 3;                % signal-peak estimator rate (1/8)

% ---- beat engine -------------------------------------------------------
c.win.pre  = 140;                 % samples before R in the 360-sample window
c.win.post = 219;                 % samples after R
c.win.len  = 360;
c.eng.ref_back  = 56;             % R-refinement reference sample = c-56
c.eng.srch_lo   = 40;             % R searched in [c-40, c-2]
c.eng.srch_hi   = 2;
c.eng.min_c     = 200;            % beats with c < 200 are skipped
c.eng.ready     = 218;            % engine starts when samples >= c+218
c.eng.rr_max    = 600;            % R-R intervals saturate at 600 samples (1.67 s)
c.eng.wait      = 640;            % no successor event after 640 samples -> process with rr_post = rr_max
c.eng.base_lo   = 38;             % baseline = mean of window[38..69]
c.eng.base_hi   = 69;
c.eng.sat_lo    = 100;            % saturation counted over window[100..180]
c.eng.sat_hi    = 180;
c.eng.sat_th    = 24;             % >=24 rail samples -> drop (AFE overload)
c.eng.flat_th   = 8;              % (max-min) < 8 codes -> drop (lead-off)
c.eng.w_sh      = 2;              % QRS edge threshold = |R-base| / 4
c.eng.on_start  = 136; c.eng.on_min = 92;     % QRS-onset scan range
c.eng.off_start = 144; c.eng.off_max = 200;   % QRS-offset scan range
c.eng.p_a = 62; c.eng.p_b = 20;   % P window = [onset-62, onset-20]
c.eng.t_a = 12; c.eng.t_len = 100;% T window = [offset+12, offset+12+100]
c.eng.t_max = 357;
c.eng.ref_sh_warm = 3;            % stochastic-median step shift (warm-up)
c.eng.ref_sh_run  = 5;            % ... steady state
c.eng.ref_warm_n  = 16;           % warm-up updates
c.eng.nf = 10;                    % number of integer features

% ---- stage-2 CNN geometry (fixed by the RTL) --------------------------
c.cnn.C1 = 8; c.cnn.C2 = 4; c.cnn.L = 120; c.cnn.H = 16; c.cnn.NV = 40;
c.cnn.pool_p = [22 37]; c.cnn.pool_q = [38 53]; c.cnn.pool_t = [60 91];

% ---- ROM layout (16-bit words) ----------------------------------------
c.rom.w1 = 0;   c.rom.b1 = 24;  c.rom.w2 = 32;  c.rom.b2 = 80;
c.rom.wf1 = 84; c.rom.bf1 = 724; c.rom.wf2 = 740; c.rom.bf2 = 788;
c.rom.len = 791;
c.tree.max_nodes = 256; c.tree.max_depth = 12;
end
