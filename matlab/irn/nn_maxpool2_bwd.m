function dX = nn_maxpool2_bwd(dY, A, sz)
% NN_MAXPOOL2_BWD  Route each gradient to the winning input of its 2x2 window.
%   sz = [C H W] of the pooled input.
C = sz(1); H = sz(2); W = sz(3);
dX = zeros(C, H, W);
for q = 1:4
    dr = mod(q - 1, 2);          % row offset of window element q
    dc = floor((q - 1) / 2);     % column offset
    dX(:, 1 + dr:2:H, 1 + dc:2:W) = dY .* (A == q);
end
end
