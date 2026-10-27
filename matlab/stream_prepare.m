function S = stream_prepare(adc, cfg, opts)
% STREAM_PREPARE  State-free front half of the accelerator on an ADC stream:
%   detector (or opts.ev_override) -> event rules -> R refinement -> beat_raw.
%   Applies exactly the skip/timeout/RR-saturation rules of run_stream_golden
%   and the RTL.  The state-dependent half is beat_decide (see decide_record).
if nargin < 3, opts = struct(); end
if isfield(opts, 'ev_override'), ev_c = opts.ev_override(:); else, ev_c = qrs_detect_golden(adc, cfg); end
N = numel(adc); e = cfg.eng; prev_c = [];
tmpl = struct('R',0,'base',0,'sat_cnt',0,'flat',false,'drop',false,'drop_sat',false,'onset',0,'offset',0, ...
              'qrs_w',0,'p_amp',0,'pr',0,'t_amp',0,'tpol',0,'rt',0,'ramp',0,'sh',0,'u',zeros(1, cfg.cnn.L));
S.ev_c = ev_c; S.k = zeros(0,1); S.c = zeros(0,1); S.rr_pre = zeros(0,1); S.rr_post = zeros(0,1);
S.raw = repmat(tmpl, 0, 1); n = 0;
for k = 1:numel(ev_c)
    c = ev_c(k);
    if k < numel(ev_c), rr_post = min(ev_c(k+1) - c, e.rr_max); can = (c + e.ready <= N);
    else, rr_post = e.rr_max; can = (c + e.wait <= N); end
    if isempty(prev_c), rr_pre = rr_post; else, rr_pre = min(c - prev_c, e.rr_max); end
    prev_c = c;
    if c < e.min_c || ~can, continue; end
    R = refine_R(adc, c, cfg); raw = beat_raw(adc, R, cfg);
    t = tmpl; fn = fieldnames(raw); for q = 1:numel(fn), t.(fn{q}) = raw.(fn{q}); end
    n = n + 1; S.raw(n,1) = t; S.k(n,1) = k; S.c(n,1) = c; S.rr_pre(n,1) = rr_pre; S.rr_post(n,1) = rr_post;
end
end
