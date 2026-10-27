function R = train_all(opts)
% TRAIN_ALL  Full experiment: DS1 -> train, DS2 -> test (De Chazal inter-patient
%   split), AFE twin in the loop, detected beats, closed-loop patient references.
%   Steps: (1) prepare records (AFE twin, detector, delineation, matching),
%   (2) iteration 1 with teacher-forced references, (3) iteration 2 with the
%   closed-loop references of the iteration-1 model, (4) tree + tau calibration
%   on held-out DS1 patients, (5) QAT CNN, (6) evaluation on DS2 for
%   {full, tree-only, CNN-only}, (7) export to xilinx/rtl/gen and RTL regression vectors.
%   opts: data_dir, results_dir, feat_mode ('rel'|'abs'), n_iter, epochs,
%         afe, ds1_ids, ds2_ids, val_ids, records_override (cell of rec structs),
%         export (true), tag.
cfg = hw_cfg(); root = fullfile(fileparts(mfilename('fullpath')), '..');
def = struct('data_dir', fullfile(root, 'data'), 'results_dir', fullfile(root, 'results'), 'feat_mode', 'rel', ...
             'n_iter', 2, 'epochs', 30, 'afe', afe_default(), 'ds1_ids', mitdb_split('ds1'), 'ds2_ids', mitdb_split('ds2'), ...
             'val_ids', [], 'records_override', [], 'export', true, 'tag', 'full', 'seed', 1, 'tree_depth', 6);
fn = fieldnames(def); for i = 1:numel(fn), if ~isfield(opts, fn{i}), opts.(fn{i}) = def.(fn{i}); end, end
if isempty(opts.val_ids), ids = opts.ds1_ids; opts.val_ids = ids(3:5:end); end
if ~exist(opts.results_dir, 'dir'), mkdir(opts.results_dir); end
tr_ids = setdiff(opts.ds1_ids, opts.val_ids);
fprintf('train records: %s\nvalidation (held-out DS1 patients): %s\ntest (DS2): %s\n', mat2str(tr_ids), mat2str(opts.val_ids), mat2str(opts.ds2_ids));
% ---------------- (1) prepare ----------------
all_ids = unique([opts.ds1_ids(:); opts.ds2_ids(:)]).'; Pm = containers.Map('KeyType', 'double', 'ValueType', 'any');
for id = all_ids
    if ~isempty(opts.records_override), rec = opts.records_override{find(cellfun(@(r) r.id, opts.records_override) == id, 1)};
    else, rec = mitdb_read(opts.data_dir, id); end
    t = tic; Pm(id) = prep_record(rec, cfg, opts.afe, struct());
    fprintf('  prepared %d: %d annotated, %d detected, det Se %.2f%% (%.1fs)\n', id, Pm(id).n_ann, numel(Pm(id).S.raw), ...
        100 * sum(Pm(id).ann_det > 0) / max(1, Pm(id).n_ann), toc(t));
end
getP = @(ids) arrayfun(@(i) {Pm(i)}, ids);  getP = @(ids) cellfun(@(i) Pm(i), num2cell(ids), 'UniformOutput', false);
Ptr = getP(tr_ids); Pva = getP(opts.val_ids); Pte = getP(opts.ds2_ids);
% ---------------- (2)/(3) iterations ----------------
model = [];
for it = 1:opts.n_iter
    fprintf('\n--- iteration %d (%s references) ---\n', it, ternary(it == 1, 'teacher-forced', 'closed-loop'));
    dopts = struct('feat_mode', opts.feat_mode, 'ref_mode', ternary(it == 1, 'label', 'pred'), 'classify', it > 1);
    Dtr = cellfun(@(P) decide_record(P, cfg, model, dopts), Ptr, 'UniformOutput', false);
    Dva = cellfun(@(P) decide_record(P, cfg, model, dopts), Pva, 'UniformOutput', false);
    [Xtr, Utr, FQtr, Ytr] = gather_set(Ptr, Dtr); [Xva, Uva, FQva, Yva] = gather_set(Pva, Dva);
    fprintf('  train beats N/S/V = %d/%d/%d   validation N/S/V = %d/%d/%d\n', sum(Ytr==0), sum(Ytr==1), sum(Ytr==2), sum(Yva==0), sum(Yva==1), sum(Yva==2));
    cnt = accumarray(Ytr + 1, 1, [3 1]); wc = min(10, (numel(Ytr) ./ (3 * max(cnt, 1))) .^ 0.5);
    T = train_tree(Xtr, Ytr, wc(Ytr + 1), opts.tree_depth, 20);
    T = tree_set_purity(T, Xva, Yva, Xtr, Ytr);
    [tau, tab] = calibrate_tau(T, Xva, Yva);
    fprintf('  tree: %d nodes, calibrated early-exit tau = %d (validation early-exit fraction %.1f%%)\n', T.nodes, tau, ...
        100 * ternary(tau < 256, tab(find(tab(:,1) == tau, 1), 2), 0));
    P0 = cnn_init(Utr, FQtr, opts.seed, cfg);
    [Pb, hist] = cnn_fit(Utr, FQtr, Ytr, Uva, FQva, Yva, P0, struct('epochs', opts.epochs, 'verbose', true)); %#ok<ASGLU>
    model = struct('tree', T, 'tau', tau, 'cnn', cnn_export(Pb));
end
% ---------------- (6) evaluation on DS2 ----------------
R.model = model; R.opts = opts;
variants = {'full', model.tau; 'tree-only', 0; 'cnn-only', 256};
for v = 1:size(variants, 1)
    m = model; m.tau = variants{v, 2};
    Dte = cellfun(@(P) decide_record(P, cfg, m, struct('feat_mode', opts.feat_mode, 'ref_mode', 'pred')), Pte, 'UniformOutput', false);
    M = score_system(Pte, Dte); R.(strrep(variants{v, 1}, '-', '_')) = M;
    print_scores(M, sprintf('DS2 [%s | features=%s | AFE %g dB, %g uVrms]', variants{v, 1}, opts.feat_mode, opts.afe.gain_dB, opts.afe.noise_uVrms));
    macs = (1 - M.early_frac) * (1 - M.drop_frac) * 9328;
    fprintf('average CNN MACs per beat: %.0f (full CNN = 9328)\n', macs);
end
save(fullfile(opts.results_dir, ['results_' opts.tag '.mat']), 'R');
% ---------------- (7) export ----------------
if opts.export && strcmp(opts.feat_mode, 'rel')
    export_rtl(model, cfg);
    fprintf('\nexported trained model to xilinx/rtl/gen (params.vh, wrom.mem, tree.mem)\n');
end
end

function v = ternary(c, a, b), if c, v = a; else, v = b; end, end
