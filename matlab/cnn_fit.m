function [Pbest, hist] = cnn_fit(Utr, FQtr, Ytr, Uva, FQva, Yva, P, opts)
% CNN_FIT  Adam + QAT.  opts: epochs, batch, lr, verbose.  Keeps the epoch with
%   the best validation F1 on the rare classes (mean of S and V, falling back
%   to accuracy when the validation set has none).
if ~isfield(opts, 'epochs'), opts.epochs = 30; end
if ~isfield(opts, 'batch'),  opts.batch = 256; end
if ~isfield(opts, 'lr'),     opts.lr = 0.02; end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
N = size(Utr, 1);
cnt = accumarray(Ytr + 1, 1, [3 1]); wc = min(10, (N ./ (3 * max(cnt, 1))) .^ 0.5);
sw_all = wc(Ytr + 1);
names = {'W1f', 'b1', 'W2f', 'b2', 'Wf1f', 'bf1', 'Wf2f', 'bf2'};
bs = [0 P.s1 0 P.s2 0 P.s3 0 P.sl];                        % bias lr scale = 2^shift
m = struct(); v = struct();
for i = 1:numel(names), m.(names{i}) = zeros(size(P.(names{i}))); v.(names{i}) = m.(names{i}); end
t = 0; best = -Inf; Pbest = P; hist = zeros(opts.epochs, 3);
for ep = 1:opts.epochs
    perm = randperm(N); tl = 0; nb = 0;
    for s = 1:opts.batch:N
        ix = perm(s:min(N, s + opts.batch - 1));
        [loss, G] = cnn_loss_grad(Utr(ix, :), FQtr(ix, :), Ytr(ix), sw_all(ix), P, true);
        t = t + 1; tl = tl + loss; nb = nb + 1;
        for i = 1:numel(names)
            nm = names{i}; g = G.(nm);
            m.(nm) = 0.9 * m.(nm) + 0.1 * g; v.(nm) = 0.999 * v.(nm) + 0.001 * g .^ 2;
            mh = m.(nm) / (1 - 0.9^t); vh = v.(nm) / (1 - 0.999^t);
            lr = opts.lr * 2^bs(i);
            P.(nm) = P.(nm) - lr * mh ./ (sqrt(vh) + 1e-8);
            if any(strcmp(nm, {'W1f', 'W2f', 'Wf1f', 'Wf2f'})), P.(nm) = min(max(P.(nm), -8.49), 7.49); end
        end
    end
    lg = cnn_fwd_batch(Uva, FQva, P, true); [~, pr] = max(lg, [], 2); pr = pr - 1;
    f1 = zeros(1, 3);
    for k = 1:3
        tp = sum(pr == k-1 & Yva == k-1); fp = sum(pr == k-1 & Yva ~= k-1); fn = sum(pr ~= k-1 & Yva == k-1);
        f1(k) = 2 * tp / max(1, 2 * tp + fp + fn);
    end
    if any(Yva > 0), score = mean(f1(2:3)); else, score = mean(pr == Yva); end
    hist(ep, :) = [tl / nb, mean(pr == Yva), score];
    if score > best, best = score; Pbest = P; end
    if opts.verbose, fprintf('  epoch %2d  loss %.4f  val acc %.3f  val F1(S,V)=%.3f %.3f\n', ep, tl / nb, mean(pr == Yva), f1(2), f1(3)); end
end
end
