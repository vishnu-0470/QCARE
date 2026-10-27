function D = decide_record(P, cfg, model, opts)
% DECIDE_RECORD  State-dependent half (features, tree, CNN, reference update)
%   over the prepared beats of one record.  opts.ref_mode = 'pred' (closed
%   loop, as in hardware) or 'label' (teacher forcing, for training data).
if nargin < 4, opts = struct(); end
if ~isfield(opts, 'ref_mode'), opts.ref_mode = 'pred'; end
n = numel(P.S.raw); st = st_init();
D.cls = 3 * ones(n, 1); D.early = zeros(n, 1); D.fv = zeros(n, cfg.eng.nf); D.drop = false(n, 1);
D.leaf_cls = zeros(n, 1); D.leaf_pur = zeros(n, 1); D.logits = zeros(n, 3);
lab_map = [0 0 1 2 -1 -1];                                   % AAMI N,S,V -> 0,1,2 ; F,Q,unmatched -> -1
for i = 1:n
    o_opts = opts;
    if strcmp(opts.ref_mode, 'label')
        l = P.lbl(i); if l == 0, o_opts.label = -1; else, o_opts.label = lab_map(l + 1); end
        if o_opts.label == 0 && l ~= 1, o_opts.label = -1; end
    end
    [o, st] = beat_decide(P.S.raw(i), P.S.rr_pre(i), P.S.rr_post(i), st, cfg, model, o_opts);
    D.cls(i) = o.cls; D.early(i) = o.early; D.fv(i, :) = o.fv; D.drop(i) = o.drop;
    D.leaf_cls(i) = o.leaf_cls; D.leaf_pur(i) = o.leaf_pur; D.logits(i, :) = o.logits;
end
end
