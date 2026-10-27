function [Pout, hist] = qat_train(P0, E, qmaps, Y, opts)
% QAT_TRAIN  Quantisation-aware training with the straight-through estimator.
%   Forward: integer model (4-bit weights, 8-bit activations, power-of-two scales).
%   Backward: float gradients computed at the quantised values, applied to the
%   latent float weights (STE).  Labels Y in 0..4.
%   opts.epochs, opts.lr, opts.verbose
%   opts.freeze_exps (default true): weight exponents are fixed at the start of
%       training.  Recomputing them each step makes the quantised weights jump
%       by a factor of 2 whenever max|W| crosses a power of two, which breaks
%       the STE.
%   opts.clip_latent (default true): latent weights are clipped to the 4-bit
%       range [-8, 7] * 2^e, keeping the STE gradient meaningful.
if ~isfield(opts, 'epochs'), opts.epochs = 20; end
if ~isfield(opts, 'lr'), opts.lr = 0.005; end
if ~isfield(opts, 'verbose'), opts.verbose = false; end
if ~isfield(opts, 'freeze_exps'), opts.freeze_exps = true; end
if ~isfield(opts, 'clip_latent'), opts.clip_latent = true; end
if ~isfield(opts, 'batch'), opts.batch = 10; end
if ~isfield(opts, 'momentum'), opts.momentum = 0.8; end
P = P0; N = size(qmaps, 1);
eWfix = []; if opts.freeze_exps, eWfix = qat_weight_exps(P0); end
V = zero_like(P);                       % momentum buffer
hist = zeros(opts.epochs, 2);
for ep = 1:opts.epochs
    tot = 0; corr = 0; perm = randperm(N);
    for s0 = 1:opts.batch:N
        ix = perm(s0:min(N, s0 + opts.batch - 1));
        Gsum = zero_like(P);
        for n = ix
            Qm = qat_build(P, E, eWfix);
            qin = reshape(qmaps(n, 1, :, :), 1, 60, 60);
            [logit, cls, C] = qnn_int_forward(qin, Qm);
            Cr = qat_real_cache(C, Qm);
            lg = double(logit) * 2^(E.blk4 + Qm.fc.eW);
            [l, dz] = nn_softmax_ce(lg, Y(n) + 1);
            tot = tot + l; corr = corr + (cls == Y(n));
            G = irn_net_backward(dz, Cr, qat_real_params(Qm));
            Gsum = add_into(Gsum, G, 1 / numel(ix));
        end
        [P, V] = sgd_momentum(P, Gsum, V, opts.lr, opts.momentum);
        if opts.clip_latent, P = clip_all(P, qat_build(P, E, eWfix)); end
    end
    hist(ep, :) = [tot / N, corr / N];
    if opts.verbose
        fprintf('  qat epoch %2d  loss %.4f  train acc %.0f%%\n', ep, tot / N, 100 * corr / N);
    end
end
Pout = P;
end

function Z = zero_like(P)
Z = P;
for a = fieldnames(P).'
    for f = fieldnames(P.(a{1})).'
        Z.(a{1}).(f{1}) = zeros(size(P.(a{1}).(f{1})));
    end
end
end

function A = add_into(A, G, w)
for a = fieldnames(A).'
    for f = fieldnames(A.(a{1})).'
        A.(a{1}).(f{1}) = A.(a{1}).(f{1}) + w * G.(a{1}).(f{1});
    end
end
end

function [P, V] = sgd_momentum(P, G, V, lr, mu)
for a = fieldnames(P).'
    for f = fieldnames(P.(a{1})).'
        V.(a{1}).(f{1}) = mu * V.(a{1}).(f{1}) - lr * G.(a{1}).(f{1});
        P.(a{1}).(f{1}) = P.(a{1}).(f{1}) + V.(a{1}).(f{1});
    end
end
end

function P = clip_all(P, Qm)
P.ds1.W = clipw(P.ds1.W, Qm.ds1.eW); P.ds2.W = clipw(P.ds2.W, Qm.ds2.eW);
P.fc.W = clipw(P.fc.W, Qm.fc.eW);
for k = 1:4
    nm = sprintf('irn%d', k); Qb = Qm.(nm);
    P.(nm).W1 = clipw(P.(nm).W1, Qb.eW1);
    P.(nm).W2 = clipw(P.(nm).W2, Qb.eW2);
    P.(nm).W3 = clipw(P.(nm).W3, Qb.eW3);
end
end

function P = latent_step(P, G, lr, Qm, clip)
for a = {'ds1', 'irn1', 'irn2', 'ds2', 'irn3', 'irn4', 'fc'}
    for f = fieldnames(P.(a{1})).'
        P.(a{1}).(f{1}) = P.(a{1}).(f{1}) - lr * G.(a{1}).(f{1});
    end
end
if clip
    P.ds1.W = clipw(P.ds1.W, Qm.ds1.eW); P.ds2.W = clipw(P.ds2.W, Qm.ds2.eW);
    P.fc.W = clipw(P.fc.W, Qm.fc.eW);
    for k = 1:4
        nm = sprintf('irn%d', k); Qb = Qm.(nm);
        P.(nm).W1 = clipw(P.(nm).W1, Qb.eW1);
        P.(nm).W2 = clipw(P.(nm).W2, Qb.eW2);
        P.(nm).W3 = clipw(P.(nm).W3, Qb.eW3);
    end
end
end

function W = clipw(W, e)
W = min(max(W, -8 * 2^e), 7 * 2^e);
end
