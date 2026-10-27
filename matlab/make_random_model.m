function model = make_random_model(seed, cfg, u_cal, fq_cal)
% MAKE_RANDOM_MODEL  Random-but-valid tree + CNN used ONLY for RTL regression
%   (exercises every arithmetic and control path). Not a trained classifier.
s = mod(double(seed) * 7919 + 1, 2147483646) + 1;
T = struct('feat', zeros(256,1), 'thr', zeros(256,1), 'left', zeros(256,1), 'right', zeros(256,1), ...
           'leaf', zeros(256,1), 'cls', zeros(256,1), 'pur', zeros(256,1));
nn = 1; stack = [1 0];                        % [node_index(1-based) depth]
while ~isempty(stack)
    id = stack(end,1); dp = stack(end,2); stack(end,:) = [];
    [r1, s] = nxt(s);
    if dp >= 5 || (dp >= 2 && r1 < 0.25) || nn > 240
        T.leaf(id) = 1; [r2, s] = nxt(s); T.cls(id) = floor(r2 * 3);
        [r3, s] = nxt(s); T.pur(id) = floor(r3 * 256);
    else
        [r2, s] = nxt(s); f = floor(r2 * 10);
        [r3, s] = nxt(s);
        if f == 6, th = 0;
        elseif f == 8, th = 15 + floor(r3 * 35);
        elseif f == 9, th = 10 + floor(r3 * 40);
        else, th = 150 + floor(r3 * 250); end
        T.feat(id) = f; T.thr(id) = th;
        T.left(id) = nn; T.right(id) = nn + 1;          % 0-based node ids
        stack = [stack; nn + 2, dp + 1; nn + 1, dp + 1]; % 1-based indices of the children
        nn = nn + 2;
    end
end
model.tree = T; model.tau = 150;
c = struct();
[c.w1, s]  = rint(s, [8 3], -8, 7);     [c.b1, s]  = rint(s, [8 1], -40, 40);
[c.w2, s]  = rint(s, [4 4 3], -8, 7);   [c.b2, s]  = rint(s, [4 1], -40, 40);
[c.wf1, s] = rint(s, [16 40], -8, 7);   [c.bf1, s] = rint(s, [16 1], -200, 200);
[c.wf2, s] = rint(s, [3 16], -8, 7);    [c.bf2, s] = rint(s, [3 1], -100, 100);
c.s1 = 4; c.s2 = 5; c.s3 = 8;
if nargin >= 4 && ~isempty(u_cal), c = cnn_calibrate_shifts(u_cal, fq_cal, c, cfg); end
model.cnn = c;
end

function [v, s] = nxt(s)
s = mod(s * 48271, 2147483647); v = s / 2147483647;
end

function [A, s] = rint(s, sz, lo, hi)
n = prod(sz); v = zeros(n, 1);
for i = 1:n
    [r, s] = nxt(s); v(i) = lo + floor(r * (hi - lo + 1));
end
A = reshape(v, sz);
end
