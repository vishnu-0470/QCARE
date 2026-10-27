% Run from anywhere: MATLAB 'run' changes folder to sim/, so locate the project root.
root = fileparts(fileparts(mfilename('fullpath')));
cd(root);
addpath(genpath(fullfile(root,'matlab')));
addpath(fullfile(root,'sim','helpers'));
% Tests for Stage 1 (CWT front end).  Run from project root.
pass = 0; fail = 0;
r = mitdb_read('data', 100);
ann = r.ann_s + 1;                      % 1-based R indices
R = ann(10); beat = r.ecg_mv(R - 140:R + 219);   % 360-sample window, R at index 141

% ---- T1: fused (one conv per scale) == direct (CWT then average pool) ----
Md = cwt_stage1(beat, struct('mode', 'direct', 'normalize', false));
Mf = cwt_stage1(beat, struct('mode', 'fused',  'normalize', false));
err = max(abs(Md(:) - Mf(:))) / max(1e-12, max(abs(Md(:))));
ok1 = err < 1e-10;
report(ok1, sprintf('T1 fused == direct pooling   (max rel err %.2e)', err)); pass = pass + double(ok1); fail = fail + double(~ok1);

% ---- T2: scale-to-frequency law: peak frequency = omega/(2*pi*j) cycles/sample ----
omega = pi / 2; ok2 = true; msg2 = '';
for j = [1 10 30 60]
    [g, ~] = cwt_kernel(j, omega);
    NF = 2^16; G = abs(fft(g, NF)); G = G(1:NF/2);
    fr = (0:NF/2 - 1) / NF;
    [~, pk] = max(G);
    f_meas = fr(pk); f_exp = omega / (2 * pi * j);
    rel = abs(f_meas - f_exp) / f_exp;
    msg2 = [msg2 sprintf(' j=%d:%.4f/%.4f', j, f_meas, f_exp)];
    ok2 = ok2 && (rel < 0.05 || abs(f_meas - f_exp) < 2 / NF);
end
report(ok2, ['T2 scale->frequency law (meas/exp cyc/sample)' msg2]); pass = pass + double(ok2); fail = fail + double(~ok2);

% ---- T3: shape and range ----
M = cwt_stage1(beat);
ok3 = isequal(size(M), [60 60]) && max(abs(M(:))) <= 1 + 1e-12 && abs(max(abs(M(:))) - 1) < 1e-12;
report(ok3, sprintf('T3 map is 60x60, normalised to [-1,1] (size %dx%d, max|M|=%.6f)', size(M,1), size(M,2), max(abs(M(:)))));
pass = pass + double(ok3); fail = fail + double(~ok3);

% ---- T4: the map localises the QRS in time (small scales respond to the R slope) ----
Mraw = cwt_stage1(beat, struct('normalize', false));
[~, cmax] = max(max(abs(Mraw(1:10, :)), [], 1));
expc = ceil(141 / 6);                       % R sample 141 -> pooled column (6 samples/column)
ok4 = abs(cmax - expc) <= 3;
report(ok4, sprintf('T4 QRS localised: peak column %d, expected ~%d', cmax, expc)); pass = pass + double(ok4); fail = fail + double(~ok4);

% ---- T5: Eq.1 segmentation: correct length, R position, zero-padding at edges ----
Rp = ann(9); Rn = ann(11);
[seg, rpos, trunc] = paper_segment(r.ecg_mv, Rp, R, Rn, 60, 360);
ok5 = numel(seg) == 360 && ~isnan(rpos) && abs(seg(rpos) - r.ecg_mv(R)) < 1e-12;
report(ok5, sprintf('T5 Eq.1 segment: length %d, R at position %d (truncated=%d)', numel(seg), rpos, trunc));
pass = pass + double(ok5); fail = fail + double(~ok5);

% ---- T6: int8 quantisation range ----
Mi = quantize_map_int8(M);
ok6 = isa(Mi, 'int8') && min(Mi(:)) >= -128 && max(Mi(:)) <= 127 && max(abs(double(Mi(:)))) == 127;
report(ok6, 'T6 8-bit quantisation of the map'); pass = pass + double(ok6); fail = fail + double(~ok6);

% ---- write maps for inspection ----
dlmwrite('sim/stage/map_rec100_beat10.txt', M, ' ');
dlmwrite('sim/stage/beat_rec100_beat10.txt', beat(:), ' ');
fprintf('\nStage 1: %d passed, %d failed\n', pass, fail);

