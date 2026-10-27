function [Y, Xp] = nn_conv2d_fwd(X, W, b, s, p)
% NN_CONV2D_FWD  2D convolution.  X [Cin H W], W [Cout Cin kh kw], b [Cout 1].
%   stride s, zero padding p.  Returns Y [Cout Ho Wo] and the padded input Xp.
[Cin, H, Wd] = size(X);
[Cout, ~, kh, kw] = size(W);
Xp = zeros(Cin, H + 2 * p, Wd + 2 * p);
Xp(:, p + 1:p + H, p + 1:p + Wd) = X;
Ho = floor((H + 2 * p - kh) / s) + 1;
Wo = floor((Wd + 2 * p - kw) / s) + 1;
Ym = zeros(Cout, Ho * Wo);
for i = 1:kh
    for j = 1:kw
        rows = i:s:i + s * (Ho - 1);
        cols = j:s:j + s * (Wo - 1);
        Xs = reshape(Xp(:, rows, cols), Cin, Ho * Wo);
        Ym = Ym + W(:, :, i, j) * Xs;
    end
end
Ym = Ym + b;
Y = reshape(Ym, Cout, Ho, Wo);
end
