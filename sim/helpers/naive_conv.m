function Y = naive_conv(X, W, b, s, p)
[Cin, H, Wd] = size(X); [Cout, ~, kh, kw] = size(W);
Xp = zeros(Cin, H + 2 * p, Wd + 2 * p); Xp(:, p + 1:p + H, p + 1:p + Wd) = X;
Ho = floor((H + 2 * p - kh) / s) + 1; Wo = floor((Wd + 2 * p - kw) / s) + 1;
Y = zeros(Cout, Ho, Wo);
for co = 1:Cout, for oh = 1:Ho, for ow = 1:Wo
    acc = b(co);
    for ci = 1:Cin, for i = 1:kh, for j = 1:kw
        acc = acc + W(co, ci, i, j) * Xp(ci, (oh - 1) * s + i, (ow - 1) * s + j);
    end, end, end
    Y(co, oh, ow) = acc;
end, end, end
end
