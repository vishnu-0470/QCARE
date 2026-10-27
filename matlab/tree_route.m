function leaf = tree_route(T, X)
% TREE_ROUTE  Leaf node (0-based) reached by every row of X.
N = size(X, 1); node = zeros(N, 1);
for step = 1:16
    act = T.leaf(node + 1) == 0; if ~any(act), break; end
    f = T.feat(node + 1); t = T.thr(node + 1);
    xv = X(sub2ind(size(X), (1:N)', f + 1));
    nxt = node; l = act & xv <= t; r = act & xv > t;
    nxt(l) = T.left(node(l) + 1); nxt(r) = T.right(node(r) + 1); node = nxt;
end
leaf = node;
end
