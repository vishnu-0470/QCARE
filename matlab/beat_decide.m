function [out, st] = beat_decide(raw, rr_pre, rr_post, st, cfg, model, opts)
% BEAT_DECIDE  State-dependent part of the beat engine: patient-referenced
%   features, stage-1 tree with confidence early exit, stage-2 CNN, and the
%   stochastic-median reference update.  Golden model of the RTL phases
%   FEATURES / TREE / CNN / UPDATE.
%     opts.feat_mode  'rel' (default, hardware) | 'abs' (ablation only)
%     opts.ref_mode   'pred' (default, closed loop) | 'label'
%     opts.label      true class 0/1/2 when ref_mode=='label'
%     opts.classify   false -> features only (training-data generation)
if nargin < 7, opts = struct(); end
if ~isfield(opts, 'feat_mode'), opts.feat_mode = 'rel'; end
if ~isfield(opts, 'ref_mode'),  opts.ref_mode  = 'pred'; end
if ~isfield(opts, 'classify'),  opts.classify  = true; end
out.R = raw.R; out.drop = raw.drop; out.cls = 3; out.early = 0;
out.fv = zeros(1, cfg.eng.nf); out.leaf_cls = 0; out.leaf_pur = 0;
out.logits = zeros(1, 3); out.rr_pre = rr_pre; out.rr_post = rr_post;
if raw.drop, return; end
% ---- features ---------------------------------------------------------
rat = @(x, ref) min(1023, floor(x * 256 / ((ref == 0) * max(x, 1) + (ref ~= 0) * ref)));
fv = zeros(1, cfg.eng.nf);
tref = double(st.pol_cnt >= 0);
if strcmp(opts.feat_mode, 'rel')
    fv(1) = rat(rr_pre,  st.rr_ref);   fv(2) = rat(rr_post, st.rr_ref);
    fv(3) = rat(raw.qrs_w, st.qrs_ref); fv(4) = rat(raw.p_amp, st.p_ref);
    fv(5) = rat(raw.t_amp, st.t_ref);   fv(6) = rat(raw.ramp,  st.r_ref);
    fv(8) = rat(raw.rt, st.rt_ref);
else
    fv(1) = min(1023, rr_pre); fv(2) = min(1023, rr_post);
    fv(3) = min(1023, raw.qrs_w * 8); fv(4) = min(1023, raw.p_amp);
    fv(5) = min(1023, raw.t_amp);     fv(6) = min(1023, raw.ramp);
    fv(8) = min(1023, raw.rt * 2);
end
fv(7)  = double(xor(raw.tpol > 0, tref > 0));
fv(9)  = min(1023, raw.qrs_w);
fv(10) = min(1023, max(0, raw.pr));
out.fv = fv;
cls = 0;
if opts.classify && ~isempty(model)
    % ---- stage-1 tree -------------------------------------------------
    tr = model.tree; node = 0; ok = false;
    for step = 0:cfg.tree.max_depth
        if tr.leaf(node + 1), ok = true; break; end
        if fv(tr.feat(node + 1) + 1) <= tr.thr(node + 1), node = tr.left(node + 1);
        else, node = tr.right(node + 1); end
    end
    if ok, lc = tr.cls(node + 1); lp = tr.pur(node + 1); else, lc = 0; lp = 0; end
    out.leaf_cls = lc; out.leaf_pur = lp;
    if lp >= model.tau
        cls = lc; out.early = 1;
    else
        fq = zeros(1, 8);
        for k = 1:8
            if k == 7, fq(k) = fv(7) * 64;
            else, fq(k) = min(max(floor((fv(k) - 256) / 4), -128), 127); end
        end
        [lg, cls] = cnn_forward_int(raw.u, fq, model.cnn, cfg);
        out.logits = lg;
    end
    out.cls = cls;
end
% ---- reference update --------------------------------------------------
st.rr_ref = ref_update(st.rr_ref, rr_pre, st.n_rr, cfg); st.n_rr = min(255, st.n_rr + 1);
if strcmp(opts.ref_mode, 'label'), isN = (opts.label == 0); else, isN = opts.classify && (cls == 0); end
if isN
    nu = st.n_mor;
    st.qrs_ref = ref_update(st.qrs_ref, raw.qrs_w, nu, cfg);
    st.p_ref   = ref_update(st.p_ref,   raw.p_amp, nu, cfg);
    st.t_ref   = ref_update(st.t_ref,   raw.t_amp, nu, cfg);
    st.r_ref   = ref_update(st.r_ref,   raw.ramp,  nu, cfg);
    st.rt_ref  = ref_update(st.rt_ref,  raw.rt,    nu, cfg);
    if raw.tpol > 0, st.pol_cnt = min(st.pol_cnt + 1, 7); else, st.pol_cnt = max(st.pol_cnt - 1, -8); end
    st.n_mor = min(255, st.n_mor + 1);
end
end
