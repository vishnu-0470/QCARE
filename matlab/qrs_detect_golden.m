function [ev_c, dbg] = qrs_detect_golden(adc, cfg)
% QRS_DETECT_GOLDEN  Bit-exact model of the streaming integer QRS detector
%   (xilinx/rtl/qrs_detector.v).  adc: integer ADC codes (0..4095).  Returns the
%   0-based coarse event index c of every detection (position of the
%   moving-window-integral maximum).  Pipeline per sample n:
%     s = x[n]-x[n-2];  e = |s|;  m = (sum of last 32 e) >> 5
%     thr = max(tmin, npk + ((spk-npk)*3 >> 3))
%     idle & refractory==0 & m>thr -> track 24 samples, keep arg-max of m
%     at the end of tracking: event, spk += (pk-spk)>>3 (first event: spk=pk)
%     idle & refractory==0 & no crossing: npk += (m-npk)>>8
if nargin < 2, cfg = hw_cfg(); end
d = cfg.det;  N = numel(adc);
x1 = adc(1); x2 = adc(1);                 % preload history with first sample
eh = zeros(1, 32); S = 0;
spk = 0; npk = 0; first = true; rc = 0; state = 0; tcnt = 0; mmax = 0; mpos = 0;
ev_c = zeros(0, 1); pks = zeros(0, 1);
for n0 = 0:N - 1
    x = adc(n0 + 1);
    s = x - x2;  e = abs(s);  x2 = x1;  x1 = x;
    ix = mod(n0, 32) + 1;
    S = S + e - eh(ix);  eh(ix) = e;
    m = floor(S / 2^d.mwi_sh);
    dd = max(spk - npk, 0);
    thr = max(d.tmin, npk + floor(dd * 3 / 8));
    rc_now = rc;
    if rc > 0, rc = rc - 1; end
    if state == 0
        if rc_now == 0
            if m > thr
                state = 1; tcnt = 0; mmax = m; mpos = n0; rc = d.refr;
            else
                npk = npk + floor((m - npk) / 2^d.npk_sh);
            end
        end
    else
        tcnt = tcnt + 1;
        if m > mmax, mmax = m; mpos = n0; end
        if tcnt == d.track
            ev_c(end + 1, 1) = mpos; pks(end + 1, 1) = mmax; %#ok<AGROW>
            if first, spk = mmax; first = false;
            else, spk = spk + floor((mmax - spk) / 2^d.spk_sh); end
            state = 0;
        end
    end
end
dbg.pk = pks;
end
