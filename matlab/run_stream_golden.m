function [beats, raws, ev_c] = run_stream_golden(adc, cfg, model, opts)
% RUN_STREAM_GOLDEN  End-to-end golden model of the accelerator on an ADC
%   code stream: streaming detector -> event FIFO -> beat engine.
%   Beat k is processed when event k+1 exists (for rr_post) OR, if no
%   successor appears within cfg.eng.wait samples, with rr_post = rr_max.
%   R-R intervals saturate at cfg.eng.rr_max.  Beats whose 360-sample
%   window does not fit in the stream are not processed - exactly the
%   behaviour of xilinx/rtl/ecg_afe_accel_top.v.
if nargin < 4, opts = struct(); end
ev_c = qrs_detect_golden(adc, cfg);
N = numel(adc); st = st_init(); prev_c = []; k_out = 0; e = cfg.eng;
beats = repmat(empty_beat(cfg), 0, 1); raws = {};
for k = 1:numel(ev_c)
    c = ev_c(k);
    if k < numel(ev_c)
        rr_post = min(ev_c(k + 1) - c, e.rr_max); can = (c + e.ready <= N);
    else
        rr_post = e.rr_max; can = (c + e.wait <= N);
    end
    if isempty(prev_c), rr_pre = rr_post; else, rr_pre = min(c - prev_c, e.rr_max); end
    prev_c = c;
    if c < e.min_c || ~can, continue; end
    R = refine_R(adc, c, cfg);
    raw = beat_raw(adc, R, cfg);
    o_opts = opts;
    if isfield(opts, 'labels_by_event') && isfield(opts, 'ref_mode') && strcmp(opts.ref_mode, 'label')
        o_opts.label = opts.labels_by_event(k);
    end
    [o, st] = beat_decide(raw, rr_pre, rr_post, st, cfg, model, o_opts);
    o.c = c; k_out = k_out + 1;
    beats(k_out, 1) = o; raws{k_out, 1} = raw; %#ok<AGROW>
end
end

function b = empty_beat(cfg)
b = struct('R', 0, 'drop', false, 'cls', 3, 'early', 0, 'fv', zeros(1, cfg.eng.nf), ...
           'leaf_cls', 0, 'leaf_pur', 0, 'logits', zeros(1, 3), 'rr_pre', 0, 'rr_post', 0, 'c', 0);
end
