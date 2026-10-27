function M = cwt_stage1(beat, opts)
% CWT_STAGE1  Stage 1 of the Quantized Inception-ResNeXt pipeline (Cao et al. 2026):
%   cosine-approximated Morlet CWT with average pooling along the translation
%   axis, producing a (scales x time) map.  Default: 60 scales x 60 positions
%   from a 360-sample beat.
%
%   opts fields (all optional):
%     scales    : scale indices j                         default 1:60
%     omega     : cosine angular frequency (rad/sample)    default pi/2
%     stride    : translation step in samples (Eq. 8: 2)   default 2
%     pool      : average-pooling width over translations  default 3
%     mode      : 'fused' (one 1D conv per scale) or 'direct' (CWT then pool)
%     normalize : divide by max|M| so M lies in [-1,1]     default true
%
%   Interpretation choices (the paper does not state these; flagged in report):
%     * omega = pi/2: keeps cos(omega*tau/j) free of aliasing at scale 1
%       (peak 90 Hz at fs=360) and reaches ~1.5 Hz at scale 60.
%     * scales 1:60, because the paper says both "scales 1-90" and "60 output
%       channels" and the 60x60 map requires 60 rows.
%     * Fig. 9 colour bar is -1..1, so the map is normalised per beat.

if nargin < 2, opts = struct(); end
opts = fill_defaults(opts);

f = double(beat(:)).';
N = numel(f);
K = floor(N / opts.stride);          % number of translations (180 for N=360)
Q = floor(K / opts.pool);            % pooled positions (60 for N=360)
S = numel(opts.scales);
M = zeros(S, Q);

for si = 1:S
    j = opts.scales(si);
    [g, tau] = cwt_kernel(j, opts.omega);
    switch opts.mode
        case 'direct'
            C = zeros(1, K);
            for k = 0:K - 1
                C(k + 1) = corr_point(f, opts.stride * k, g, tau);
            end
            P = reshape(C(1:Q * opts.pool), opts.pool, Q);
            M(si, :) = mean(P, 1);
        case 'fused'
            [H, tauH] = fused_kernel(g, tau, opts.stride, opts.pool);
            for q = 0:Q - 1
                M(si, q + 1) = corr_point(f, opts.stride * opts.pool * q, H, tauH);
            end
        otherwise
            error('cwt_stage1: mode must be fused or direct');
    end
end

if opts.normalize
    mx = max(abs(M(:)));
    if mx > 0, M = M / mx; end
end
end

function v = corr_point(f, m, g, tau)
% sum_n f[n] * g[n - m]   for n = 0..N-1 (0-based sample index n)
N = numel(f);
n = m + tau;
ok = (n >= 0) & (n <= N - 1);
v = sum(f(n(ok) + 1) .* g(ok));
end

function [H, tauH] = fused_kernel(g, tau, stride, pool)
% Fold the pooling into the kernel:  H(t) = (1/pool) * sum_r g(t - stride*r)
a = min(tau); b = max(tau);
tauH = a:(b + stride * (pool - 1));
H = zeros(size(tauH));
for r = 0:pool - 1
    idx = tauH - stride * r;
    in = (idx >= a) & (idx <= b);
    H(in) = H(in) + g(idx(in) - a + 1);
end
H = H / pool;
end

function o = fill_defaults(o)
if ~isfield(o, 'scales'),    o.scales = 1:60; end
if ~isfield(o, 'omega'),     o.omega = pi / 2; end
if ~isfield(o, 'stride'),    o.stride = 2; end
if ~isfield(o, 'pool'),      o.pool = 3; end
if ~isfield(o, 'mode'),      o.mode = 'fused'; end
if ~isfield(o, 'normalize'), o.normalize = true; end
end
