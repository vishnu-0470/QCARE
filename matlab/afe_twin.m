function [adc, info] = afe_twin(ecg_mv, p)
% AFE_TWIN  Behavioural twin of the CCIA + gain + LPF + 12-bit ADC chain.
%   adc = afe_twin(ecg_mv, p)  ->  column of integer codes 0..2^bits-1 at fs.
%
%   Signal chain (simulated at 4x oversampling, then sampled by the ADC):
%     x_in (mV) + noise + mains + electrode offset
%       -> DC-servo high-pass (corner p.f_hp), gain G, hard clip at the rails
%       -> 2nd-order Butterworth low-pass (p.f_lp)  -> sample-and-hold -> ADC
%   The high-pass is a servo loop driven by the CLIPPED output, so an overload
%   recovers more slowly than a linear high-pass (p.rec_slow multiplies the
%   servo time constant while the output is clipped).
%
%   IMPORTANT: every default below is a placeholder until it is calibrated
%   against Spectre with sim/validate_afe_twin.m (small-signal AC, a linear
%   10 Hz transient, an offset step and a noise run). The rails default to
%   the min/max of the overdriven transient in clean_out.csv.
if nargin < 2, p = struct(); end
p = setdef(p, 'fs',        360);
p = setdef(p, 'gain_dB',   38.0);
p = setdef(p, 'f_hp',      0.5);
p = setdef(p, 'f_lp',      150);
p = setdef(p, 'v_clip_pos', 0.6887);
p = setdef(p, 'v_clip_neg', -0.7945);
p = setdef(p, 'bits',      12);
p = setdef(p, 'noise_uVrms', 0);     % input-referred, integrated over f_lp
p = setdef(p, 'mains_uV',  0);       % input-referred mains pickup amplitude
p = setdef(p, 'mains_hz',  50);
p = setdef(p, 'offset_mV', 0);       % electrode DC offset
p = setdef(p, 'rec_slow',  1);       % servo slow-down while clipped (>=1)
p = setdef(p, 'seed',      1);
p = setdef(p, 'os',        4);       % oversampling factor

G   = 10^(p.gain_dB / 20);
fso = p.fs * p.os;  Ts = 1 / fso;
x   = ecg_mv(:) * 1e-3 + p.offset_mV * 1e-3;            % volts at the input
N   = numel(x);
% --- 4x linear interpolation of the (band-limited) input ----------------
t0 = (0:N - 1).' / p.fs;  t1 = (0:(N - 1) * p.os).' / fso;
xo = interp1(t0, x, t1, 'linear');
% --- additive impairments ----------------------------------------------
if p.noise_uVrms > 0
    sig = p.noise_uVrms * 1e-6 * sqrt((fso / 2) / p.f_lp);
    xo = xo + sig * randn_seeded(numel(xo), p.seed);
end
if p.mains_uV > 0
    xo = xo + p.mains_uV * 1e-6 * sin(2 * pi * p.mains_hz * t1);
end
% --- DC-servo high-pass + gain + hard clip ------------------------------
tau = 1 / (2 * pi * p.f_hp);
M = numel(xo);  y = zeros(M, 1);  l = xo(1);  sat = false(M, 1);
for n = 1:M
    yv = G * (xo(n) - l);
    yc = min(max(yv, p.v_clip_neg), p.v_clip_pos);
    sat(n) = (yc ~= yv);
    y(n) = yc;
    k = Ts / (tau * (1 + (p.rec_slow - 1) * sat(n)));
    l = l + (yc / G) * k;
end
% --- 2nd-order Butterworth low-pass (bilinear), unity DC gain ------------
K = tan(pi * p.f_lp / fso);  q = 1 / sqrt(2);
nrm = 1 / (1 + K / q + K * K);
b = [K * K, 2 * K * K, K * K] * nrm;
a = [1, 2 * (K * K - 1) * nrm, (1 - K / q + K * K) * nrm];
y = filter_df2(b, a, y);
% --- sample-and-hold + ADC ----------------------------------------------
ys = y(1:p.os:end);  ys = ys(1:N);
full = 2^p.bits - 1;
adc = round((ys - p.v_clip_neg) ./ (p.v_clip_pos - p.v_clip_neg) * full);
adc = min(max(adc, 0), full);
info.gain_vv = G;  info.lsb_in_uV = (p.v_clip_pos - p.v_clip_neg) / full / G * 1e6;
info.sat_frac = mean(sat);  info.params = p;
end

function p = setdef(p, f, v)
if ~isfield(p, f), p.(f) = v; end
end

function y = filter_df2(b, a, x)
y = zeros(size(x)); z1 = 0; z2 = 0;
for n = 1:numel(x)
    v = x(n) - a(2) * z1 - a(3) * z2;
    y(n) = b(1) * v + b(2) * z1 + b(3) * z2;
    z2 = z1; z1 = v;
end
end

function r = randn_seeded(n, seed)
% Deterministic Gaussian noise, identical in MATLAB and Octave:
% Park-Miller minstd LCG (exact in double precision) + Box-Muller.
s = mod(double(seed) * 7919 + 1, 2147483646) + 1;
m = 2 * ceil(n / 2); u = zeros(m, 1);
for i = 1:m
    s = mod(s * 48271, 2147483647);
    u(i) = s / 2147483647;
end
r1 = sqrt(-2 * log(u(1:2:end))) .* cos(2 * pi * u(2:2:end));
r2 = sqrt(-2 * log(u(1:2:end))) .* sin(2 * pi * u(2:2:end));
r = [r1; r2]; r = r(1:n);
end
