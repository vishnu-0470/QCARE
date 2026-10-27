% Run from anywhere: MATLAB 'run' changes folder to sim/, so locate the project root.
root = fileparts(fileparts(mfilename('fullpath')));
cd(root);
addpath(genpath(fullfile(root,'matlab')));
addpath(fullfile(root,'sim','helpers'));
% Tests for Stage 3 (4-bit weight / 8-bit activation QAT).  Run from project root.
pass = 0; fail = 0;
if exist('OCTAVE_VERSION', 'builtin'), randn('state', 3); rand('state', 3); else, rng(3); end

% ---- T1: Eq. 3 weight scale and 4-bit range ----
okT1 = true; worst = 0;
for t = 1:200
    W = randn(4, 5) * 10^(randi([-3 1]));
    [q, e] = qat_weight_quant(W);
    km = max(abs(W(:)));
    r = floor(log2(km));
    okT1 = okT1 && (2^r < km && km < 2^(r + 1)) && e == r - 2 && all(q(:) >= -8 & q(:) <= 7);
    worst = max(worst, max(max(abs(W / 2^e - q))) );
    unsat = abs(W / 2^e) <= 7.5;                  % Eq.3 range saturates above 7.5 LSB
    okT1 = okT1 && all(all(abs(W(unsat) / 2^e - q(unsat)) <= 0.5 + 1e-12));
end
report(okT1, 'T1 Eq.3 scale e=r-2, q in [-8,7], rounding error <= LSB/2 for unsaturated weights');
fprintf('       note: Eq.3 max code is 7*2^(r-2) < 2^(r+1), so the top 1/16 of the range saturates\n');
pass = pass + double(okT1); fail = fail + double(~okT1);

% ---- T2: round-half-up requantisation ----
okT2 = qat_requant(3, 1) == 2 && qat_requant(-3, 1) == -1 && qat_requant(5, -2) == 20 && ...
       qat_requant(7, 0) == 7 && qat_requant(4, 2) == 1 && qat_requant(6, 2) == 2;
report(okT2, 'T2 requantisation rounds half-up and handles left shifts');
pass = pass + double(okT2); fail = fail + double(~okT2);

% ---- build a quantised model: calibrate on DS1 record 119 (waveform only), no labels needed ----
r119 = mitdb_read('data', 119);
P = irn_net_init(1);
cal_maps = zeros(6, 1, 60, 60); q_cal = zeros(6, 1, 60, 60);
for k = 1:6
    beat = r119.ecg_mv(1000 + (k - 1) * 360 : 1000 + k * 360 - 1);
    M = cwt_stage1(beat);
    cal_maps(k, 1, :, :) = M;
    q_cal(k, 1, :, :) = floor(M * 128 + 0.5);
end
q_cal = min(max(q_cal, -128), 127);
E = qat_calibrate(P, q_cal);
Qm = qat_build(P, E);
fprintf('       exponents: ds1 %d | blk1 %d | blk2 %d | ds2 %d | blk3 %d | blk4 %d\n', ...
        E.ds1, E.blk1, E.blk2, E.ds2, E.blk3, E.blk4);

% ---- T3: integer forward: ranges and integrality ----
okT3 = true; maxacc = 0;
for k = 1:6
    qin = reshape(q_cal(k, 1, :, :), 1, 60, 60);
    [logit, cls, C] = qnn_int_forward(qin, Qm);
    okT3 = okT3 && all(logit == round(logit)) && all(C.U2(:) >= 0 & C.U2(:) <= 127) && ...
           all(C.U4(:) >= 0 & C.U4(:) <= 127) && cls >= 0 && cls <= 4;
    maxacc = max(maxacc, max(abs(C.acc_ds1(:))));
end
okT3 = okT3 && maxacc < 2^23;
report(okT3, sprintf('T3 integer forward: integral logits, activations in [0,127], |acc| < 2^23 (max %.0f)', maxacc));
pass = pass + double(okT3); fail = fail + double(~okT3);

% ---- T4: QAT learns: memorise 10 distinct random quantised maps ----
Xr = randn(10, 1, 60, 60); Xr = max(min(Xr, 1), -1);
qr = min(max(floor(Xr * 128 + 0.5), -128), 127);
Yr = mod((0:9)', 5);
Er = qat_calibrate(P, qr);
[Pq, hist] = qat_train(P, Er, qr, Yr, struct('epochs', 40, 'lr', 0.005));
okT4 = (hist(end, 2) == 1) && (hist(end, 1) < 0.5 * hist(1, 1));
report(okT4, sprintf('T4 QAT memorises 10 random maps: loss %.3f -> %.3f, train acc %.0f%% (40 epochs)', ...
    hist(1, 1), hist(end, 1), 100 * hist(end, 2)));
fprintf('       KNOWN LIMITATION: quantisation error on this network is ~2.9x the logit spread\n');
fprintf('       between inputs (sim/t_stage3_fidelity.m), so 4W/8A QAT does not memorise here.\n');
pass = pass + double(okT4); fail = fail + double(~okT4);

% ---- T5: run the trained integer model on real beats of record 100 (DS2) ----
%   Record 100 is DS2 (test set).  This is a run-through, not a result.
r100 = mitdb_read('data', 100); ann = r100.ann_s + 1;
Qq = qat_build(Pq, Er); cnt = zeros(1, 5);
for k = 1:30
    R = ann(k + 2); M = cwt_stage1(r100.ecg_mv(R - 140:R + 219));
    qin = reshape(min(max(floor(M * 128 + 0.5), -128), 127), 1, 60, 60);
    [~, cls] = qnn_int_forward(qin, Qq); cnt(cls + 1) = cnt(cls + 1) + 1;
end
okT5 = sum(cnt) == 30;
report(okT5, sprintf('T5 integer model runs on 30 DS2 beats (predicted N,S,V,F,Q = %s) - not a result', mat2str(cnt)));
pass = pass + double(okT5); fail = fail + double(~okT5);

fprintf('\nStage 3: %d passed, %d failed\n', pass, fail);
