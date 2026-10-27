function T = tree_set_purity(T, Xv, Yv, Xt, Yt)
% TREE_SET_PURITY  purity (0..255) of each leaf = Laplace-smoothed fraction of
%   the leaf's class among HELD-OUT beats (>=10 held-out beats), otherwise a
%   conservative 0.9x of the training estimate.
lv = tree_route(T, Xv); lt = tree_route(T, Xt);
for id = find(T.leaf(1:T.nodes)).' - 1
    c = T.cls(id + 1); mv = lv == id; mt = lt == id;
    if sum(mv) >= 10, p = (sum(Yv(mv) == c) + 1) / (sum(mv) + 2);
    else, p = 0.9 * (sum(Yt(mt) == c) + 1) / (sum(mt) + 2); end
    T.pur(id + 1) = min(255, floor(255 * p));
end
end
