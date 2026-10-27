% Run from anywhere: MATLAB 'run' changes folder to sim/, so locate the project root.
root = fileparts(fileparts(mfilename('fullpath')));
cd(root);
addpath(genpath(fullfile(root,'matlab')));
addpath(fullfile(root,'sim','helpers'));
% Tests for Stage 2 (Inception-ResNeXt network).  Run from project root.
pass = 0; fail = 0;

% ---- real CWT maps from Stage 1 (record 100, first 12 annotated beats) ----
r = mitdb_read('data', 100); ann = r.ann_s + 1;
maps = zeros(12, 1, 60, 60);
for k = 1:12
    R = ann(k + 2); beat = r.ecg_mv(R - 140:R + 219);
    maps(k, 1, :, :) = cwt_stage1(beat);
end
X1 = reshape(maps(1, 1, :, :), 1, 60, 60);

% ---- T1: parameter count and shape chain ----
P = irn_net_init(1);
[logits, cache] = irn_net_forward(X1, P);
ok = isequal(size(cache.z1), [8 20 20]) && isequal(size(cache.p), [8 10 10]) && ...
     isequal(size(cache.z2), [16 4 4]) && isequal(size(logits), [5 1]);
report(ok, sprintf('T1 shapes: DS1 %s | pool %s | DS2 %s | logits %s', ...
    mat2str(size(cache.z1)), mat2str(size(cache.p)), mat2str(size(cache.z2)), mat2str(size(logits))));
pass = pass + double(ok); fail = fail + double(~ok);
np = irn_net_count(P);
fprintf('       trainable parameters (float): %d  -> %.1f kbit at 4 bits\n', np, np * 4 / 1000);

% ---- T2: convolution matches a naive reference implementation ----
Xr = randn(2, 7, 9); Wr = randn(3, 2, 3, 3); br = randn(3, 1);
Yf = nn_conv2d_fwd(Xr, Wr, br, 2, 1);
Yn = naive_conv(Xr, Wr, br, 2, 1);
ok = max(abs(Yf(:) - Yn(:))) < 1e-12;
report(ok, sprintf('T2 conv2d (stride 2, pad 1) equals naive reference (max err %.2e)', max(abs(Yf(:) - Yn(:)))));
pass = pass + double(ok); fail = fail + double(~ok);

% ---- T3: gradient check of the full network (finite differences) ----
[loss0, G] = loss_and_grads(X1, 3, P);
names = flat_names(P);
worst = 0; checked = 0; h = 1e-5; rng_s = 7;

for n = 1:numel(names)
    p = names{n};
    val = get_field(P, p);
    gval = get_field(G, p);
    idxs = unique(round(linspace(1, numel(val), min(numel(val), 3))));
    for q = idxs
        Pp = set_field(P, p, q,  h); lp = loss_only(X1, 3, Pp);
        Pm = set_field(P, p, q, -h); lm = loss_only(X1, 3, Pm);
        num = (lp - lm) / (2 * h); ana = gval(q);
        rel = abs(num - ana) / max(1e-7, abs(num) + abs(ana));
        worst = max(worst, rel); checked = checked + 1;
    end
end
ok = worst < 1e-4;
report(ok, sprintf('T3 gradient check over %d parameter entries: worst relative error %.2e', checked, worst));
pass = pass + double(ok); fail = fail + double(~ok);

% ---- T4: learning / capacity check on DISTINCT random maps (memorise 10 labels) ----
%   Standard sanity test for an optimiser + backward pass.  Not a classification result.
if exist('OCTAVE_VERSION', 'builtin'), randn('state', 11); else, rng(11); end
Xs = randn(10, 1, 60, 60); Y = mod((0:9)', 5) + 1;
Ptr = irn_net_init(2); lr = 0.005; acc_end = 0; loss_end = Inf; loss_start = NaN;
for ep = 1:60
    tot = 0; corr = 0;
    for s_ = 1:10
        [lg, c] = irn_net_forward(reshape(Xs(s_, :, :, :), 1, 60, 60), Ptr);
        [l, dz] = nn_softmax_ce(lg, Y(s_));
        tot = tot + l; [~, pk] = max(lg); corr = corr + (pk == Y(s_));
        Ptr = sgd_step(Ptr, irn_net_backward(dz, c, Ptr), lr);
    end
    if ep == 1, loss_start = tot / 10; end
    loss_end = tot / 10; acc_end = corr / 10;
end
ok = (acc_end == 1) && (loss_end < 0.1);
report(ok, sprintf('T4 memorises 10 distinct random maps: loss %.3f -> %.3f, accuracy %.0f%% (60 epochs)', ...
    loss_start, loss_end, 100 * acc_end));
pass = pass + double(ok); fail = fail + double(~ok);

% ---- INFO (not a pass/fail test): same learner on 10 REAL maps, synthetic labels ----
%   The real maps of consecutive beats are almost identical after normalisation
%   (pixelwise std ~0.04), so cyclic labels give cancelling gradients and the
%   loss stays near ln(5) in 80 epochs.  This is a property of the data and
%   the GAP-averaged features, not a gradient error (T3 passes).
fprintf('INFO  real-beat synthetic-label test: see sim/t_stage2_lr.m (loss stays near ln5=1.609)\n');

fprintf('\nStage 2: %d passed, %d failed\n', pass, fail);

