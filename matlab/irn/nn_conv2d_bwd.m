function [dX, dW, db] = nn_conv2d_bwd(dY, X, W, s, p)
% NN_CONV2D_BWD  Gradients of nn_conv2d_fwd.  Returns dX [Cin H W], dW, db [Cout 1].
[Cin, H, Wd] = size(X);
[Cout, ~, kh, kw] = size(W);
[~, Ho, Wo] = size(dY);
Xp = zeros(Cin, H + 2 * p, Wd + 2 * p);
Xp(:, p + 1:p + H, p + 1:p + Wd) = X;
dXp = zeros(size(Xp));
dYm = reshape(dY, Cout, Ho * Wo);
dW = zeros(size(W));
for i = 1:kh
    for j = 1:kw
        rows = i:s:i + s * (Ho - 1);
        cols = j:s:j + s * (Wo - 1);
        Xs = reshape(Xp(:, rows, cols), Cin, Ho * Wo);
        dW(:, :, i, j) = dYm * Xs.';
        dXs = W(:, :, i, j).' * dYm;
        dXp(:, rows, cols) = dXp(:, rows, cols) + reshape(dXs, Cin, Ho, Wo);
    end
end
db = sum(dYm, 2);
dX = dXp(:, p + 1:p + H, p + 1:p + Wd);
end
