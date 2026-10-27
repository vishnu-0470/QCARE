function raw = beat_raw(adc, R, cfg)
% BEAT_RAW  State-free part of the beat engine (golden model of the RTL
%   phases COPY / GATE / DELINEATE / NORMALISE).  adc: integer ADC codes,
%   R: 0-based index of the (refined) R peak.  All arithmetic is integer.
e = cfg.eng;  L = cfg.win.len;
n0 = R - cfg.win.pre;
bw = adc(n0 + 1 : n0 + L);  bw = bw(:);              % bw(j+1) = window[j]
raw.R = R;
base = floor(sum(bw(e.base_lo + 1 : e.base_hi + 1)) / 32);
sat  = sum(bw(e.sat_lo + 1 : e.sat_hi + 1) >= 4094 | bw(e.sat_lo + 1 : e.sat_hi + 1) <= 1);
flat = (max(bw) - min(bw)) < e.flat_th;
raw.base = base; raw.sat_cnt = sat; raw.flat = flat;
raw.drop = (sat >= e.sat_th) || flat;
raw.drop_sat = sat >= e.sat_th;
if raw.drop, return; end
% ---- QRS onset/offset by amplitude threshold --------------------------
%   th = max(|R-base| >> w_sh, 4).  Onset = largest j in [on_min,on_start]
%   with two consecutive samples below th; offset = smallest j in
%   [off_start,off_max] likewise.  (A slope rule stops at the flat apex of
%   a wide PVC and under-estimates its width, so it is not used.)
ramp = max(1, abs(bw(cfg.win.pre + 1) - base));
th = max(floor(ramp / 2^e.w_sh), 4);
ad = @(j) abs(bw(j + 1) - base);
onset = e.on_min;
for j = e.on_start:-1:e.on_min
    if ad(j) < th && ad(j + 1) < th, onset = j; break; end
end
offset = e.off_max;
for j = e.off_start:e.off_max
    if ad(j) < th && ad(j - 1) < th, offset = j; break; end
end
raw.onset = onset; raw.offset = offset; raw.qrs_w = offset - onset;
% ---- P wave -----------------------------------------------------------
m5 = @(j) bw(j - 1) + bw(j) + bw(j + 1) + bw(j + 2) + bw(j + 3);  % window[j-2..j+2]
pw0 = onset - e.p_a; pw1 = onset - e.p_b;
e0 = m5(pw0); e1 = m5(pw1); best = 0; jp = pw0;
for j = pw0:pw1
    dv = abs(2 * m5(j) - e0 - e1);
    if dv > best, best = dv; jp = j; end
end
raw.p_amp = floor(best / 8); raw.pr = onset - jp;
% ---- T wave -----------------------------------------------------------
tw0 = offset + e.t_a; tw1 = min(tw0 + e.t_len, e.t_max);
b5 = 5 * base; best = 0; jt = tw0;
for j = tw0:tw1
    dv = abs(m5(j) - b5);
    if dv > best, best = dv; jt = j; end
end
raw.t_amp = floor(best / 8); raw.tpol = double(m5(jt) > b5); raw.rt = jt - cfg.win.pre;
raw.ramp = ramp;
% ---- int8 normalisation and 3:1 down-sampling for the CNN -------------
wc = bw - base;  mx = max(abs(wc)); sh = 0;
while floor(mx / 2^sh) > 127, sh = sh + 1; end
wv = min(max(floor(wc / 2^sh), -128), 127);
u = floor((wv(1:3:end) + wv(2:3:end) + wv(3:3:end)) / 4);
raw.sh = sh; raw.u = u(:).';
end
