function P = cnn_init(U, FQ, seed, cfg)
% CNN_INIT  Latent (float) weights, calibrated requantisation shifts s1..s3 and
%   the training-only logit temperature sl.
if exist('OCTAVE_VERSION', 'builtin'), randn('state', seed); else, rng(seed); end
P.W1f = 2.5 * randn(8, 3); P.W2f = 2.0 * randn(4, 4, 3);
P.Wf1f = 1.5 * randn(16, 40); P.Wf2f = 2.0 * randn(3, 16);
P.b1 = zeros(8, 1); P.b2 = zeros(4, 1); P.bf1 = zeros(16, 1); P.bf2 = zeros(3, 1);
q = @(w) min(max(round(w), -8), 7);
c = struct('w1', q(P.W1f), 'b1', P.b1, 'w2', q(P.W2f), 'b2', P.b2, 'wf1', q(P.Wf1f), 'bf1', P.bf1, ...
           'wf2', q(P.Wf2f), 'bf2', P.bf2, 's1', 0, 's2', 0, 's3', 0);
c = cnn_calibrate_shifts(U, FQ, c, cfg);
P.s1 = c.s1; P.s2 = c.s2; P.s3 = c.s3; P.sl = 0;
n = min(size(U, 1), 500); lg = cnn_fwd_batch(U(1:n, :), FQ(1:n, :), P, true);
P.sl = max(0, round(log2(max(std(lg(:)), 1) / 3)));
end
