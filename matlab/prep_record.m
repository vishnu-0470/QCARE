function P = prep_record(rec, cfg, afe_p, opts)
% PREP_RECORD  Run the AFE twin + state-free front half on one record and
%   match the detected beats to the annotations.
%   opts.annot_timing=true replaces the detector by annotation timing
%   (events at annotation+11 samples, the median detector lag).
if nargin < 4, opts = struct(); end
adc = afe_twin(rec.ecg_mv, afe_p);
if isfield(opts, 'annot_timing') && opts.annot_timing, opts.ev_override = rec.ann_s + 11; end
S = stream_prepare(adc, cfg, opts);
P.id = rec.id; P.S = S; P.ann_s = rec.ann_s; P.ann_cls = rec.ann_cls; P.n_ann = numel(rec.ann_s);
if isempty(S.raw), R = zeros(0,1); else, R = [S.raw.R]'; end
[P.lbl, P.ann_det] = match_beats(R, rec.ann_s, rec.ann_cls, 54);
P.n_events = numel(S.ev_c);
end
