1;
function s = ifelse_(c, a, b), if c, s = a; else, s = b; end, end
% Unit tests for the CNN trainer: (1) STE-free gradient check in float mode,
% (2) integer-exactness of the vectorised QAT forward vs cnn_forward_int,
% (3) a short learning run on real beats of record 100 (smoke test).
addpath('matlab'); cfg = hw_cfg();
if exist('OCTAVE_VERSION', 'builtin'), randn('state', 3); rand('state', 3); else, rng(3); end
% ---- build a small real dataset (record 100, 2 min) ----------------------
r = mitdb_read('data', 100); x = r.ecg_mv(1:360*180);
adc = afe_twin(x, struct('noise_uVrms', 3));
S = stream_prepare(adc, cfg, struct());
P0 = struct('S', S, 'lbl', zeros(numel(S.raw), 1));
[l, ~] = match_beats([S.raw.R]', r.ann_s(r.ann_s < numel(x)), r.ann_cls(r.ann_s < numel(x)), 54); P0.lbl = l;
D = decide_record(P0, cfg, [], struct('classify', false, 'ref_mode', 'label'));
ok = ~D.drop & P0.lbl >= 1 & P0.lbl <= 3;
U = reshape([S.raw(ok).u], 120, []).'; FV = D.fv(ok, :); Y = P0.lbl(ok) - 1;
FQ = zeros(size(U, 1), 8);
for j = 1:8, if j == 7, FQ(:, j) = FV(:, 7) * 64; else, FQ(:, j) = min(max(floor((FV(:, j) - 256) / 4), -128), 127); end, end
fprintf('dataset: %d beats (N=%d S=%d V=%d)\n', numel(Y), sum(Y==0), sum(Y==1), sum(Y==2));
P = cnn_init(U, FQ, 5, cfg); fprintf('init shifts s1=%d s2=%d s3=%d sl=%d\n', P.s1, P.s2, P.s3, P.sl);
% ---- (2) integer exactness ----------------------------------------------
n = 60; idx = round(linspace(1, size(U, 1), n)); lg = cnn_fwd_batch(U(idx, :), FQ(idx, :), P, true);
c = cnn_export(P); mism = 0;
for q = 1:n
    [l2, cls2] = cnn_forward_int(U(idx(q), :), FQ(idx(q), :), c, cfg);
    if any(l2 ~= lg(q, :)), mism = mism + 1; end
end
fprintf('TEST 2  vectorised QAT forward == integer golden: %d/%d mismatching beats  -> %s\n', mism, n, ifelse_(mism == 0, 'PASS', 'FAIL'));
% ---- (1) gradient check (float mode) ------------------------------------
Ub = U(idx(1:24), :) / 4; FQb = FQ(idx(1:24), :) / 8; Yb = Y(idx(1:24)); sw = ones(24, 1);
Pf = P; Pf.sl = 4; Pf.b1 = 0.3 * randn(8, 1); Pf.b2 = 0.3 * randn(4, 1); Pf.bf1 = 3 * randn(16, 1); Pf.bf2 = randn(3, 1);
[l0, G] = cnn_loss_grad(Ub, FQb, Yb, sw, Pf, false);
names = {'W1f', 'W2f', 'Wf1f', 'Wf2f', 'b1', 'b2', 'bf1', 'bf2'}; worst = 0; h = 1e-4;
for i = 1:numel(names)
    nm = names{i}; sz = numel(Pf.(nm)); pick = unique(round(linspace(1, sz, min(sz, 6))));
    for p = pick
        Pp = Pf; Pp.(nm)(p) = Pp.(nm)(p) + h; lp = cnn_loss_grad(Ub, FQb, Yb, sw, Pp, false);
        Pm = Pf; Pm.(nm)(p) = Pm.(nm)(p) - h; lm = cnn_loss_grad(Ub, FQb, Yb, sw, Pm, false);
        num = (lp - lm) / (2 * h); ana = G.(nm)(p);
        rel = abs(num - ana) / max(1e-6, abs(num) + abs(ana)); worst = max(worst, rel);
    end
end
fprintf('TEST 1  gradient check: worst relative error = %.2e  -> %s\n', worst, ifelse_(worst < 1e-3, 'PASS', 'FAIL'));
% ---- (3) learning smoke test --------------------------------------------
perm = randperm(size(U, 1)); ntr = round(0.7 * numel(perm)); tr = perm(1:ntr); va = perm(ntr+1:end);
[Pb, hist] = cnn_fit(U(tr, :), FQ(tr, :), Y(tr), U(va, :), FQ(va, :), Y(va), P, struct('epochs', 8, 'verbose', false));
fprintf('TEST 3  loss per epoch: %s\n', sprintf('%.3f ', hist(:, 1)));
fprintf('        val accuracy per epoch: %s\n', sprintf('%.3f ', hist(:, 2)));
fprintf('        loss decreased: %s\n', ifelse_(hist(end, 1) < hist(1, 1), 'PASS', 'FAIL'));
