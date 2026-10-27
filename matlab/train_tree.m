function T = train_tree(X, Y, W, max_depth, min_leaf)
% TRAIN_TREE  Weighted CART (Gini) on integer features -> hardware tree.
%   X [N x nf] integers, Y in {0,1,2}, W sample weights.  Split rule is
%   x(feat) <= thr -> left (the RTL comparison).  Leaves get class = weighted
%   majority; purity is set later from held-out data (tree_set_purity).
N = size(X, 1); nf = size(X, 2);
T = struct('feat', zeros(256,1), 'thr', zeros(256,1), 'left', zeros(256,1), 'right', zeros(256,1), ...
           'leaf', ones(256,1), 'cls', zeros(256,1), 'pur', zeros(256,1), 'nodes', 1);
queue = {struct('id', 0, 'idx', (1:N)', 'depth', 0)}; next_id = 1;
while ~isempty(queue)
    q = queue{1}; queue(1) = []; idx = q.idx; n = numel(idx);
    Wc = accumarray(Y(idx) + 1, W(idx), [3 1]); tot = sum(Wc); [mw, maj] = max(Wc);
    T.cls(q.id + 1) = maj - 1; T.leaf(q.id + 1) = 1;
    if q.depth >= max_depth || n < 2 * min_leaf || mw / tot > 0.999 || next_id > 250, continue; end
    Gp = 1 - sum((Wc / tot).^2); best = 1e-7; bf = -1; bt = 0;
    for f = 1:nf
        [xs, ord] = sort(X(idx, f)); ys = Y(idx(ord)); ws = W(idx(ord));
        oh = zeros(n, 3); oh(sub2ind([n 3], (1:n)', ys + 1)) = ws; cl = cumsum(oh, 1);
        v = find(xs(1:end-1) < xs(2:end)); v = v(v >= min_leaf & v <= n - min_leaf);
        if isempty(v), continue; end
        L = cl(v, :); Rr = repmat(Wc.', numel(v), 1) - L; wl = sum(L, 2); wr = sum(Rr, 2);
        gl = 1 - sum((L ./ wl).^2, 2); gr = 1 - sum((Rr ./ wr).^2, 2);
        gain = Gp - (wl .* gl + wr .* gr) / tot; [g, qq] = max(gain);
        if g > best, best = g; bf = f - 1; bt = floor((xs(v(qq)) + xs(v(qq) + 1)) / 2); end
    end
    if bf < 0, continue; end
    goleft = X(idx, bf + 1) <= bt;
    T.leaf(q.id + 1) = 0; T.feat(q.id + 1) = bf; T.thr(q.id + 1) = bt;
    T.left(q.id + 1) = next_id; T.right(q.id + 1) = next_id + 1;
    queue{end + 1} = struct('id', next_id, 'idx', idx(goleft), 'depth', q.depth + 1);     %#ok<AGROW>
    queue{end + 1} = struct('id', next_id + 1, 'idx', idx(~goleft), 'depth', q.depth + 1); %#ok<AGROW>
    next_id = next_id + 2;
end
T.nodes = next_id;
end
