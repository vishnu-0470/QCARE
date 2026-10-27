% Smoke test of the whole training/evaluation flow on LOCAL data only.
% Record 100 has real labels; record 119's waveform has no annotation file, so
% for THIS CODE TEST ONLY it receives heuristic labels (wide QRS -> V). The
% numbers printed here are meaningless as science - they only prove the flow runs.
addpath('matlab'); cfg = hw_cfg();
r100 = mitdb_read('data', 100); r100.ecg_mv = r100.ecg_mv(1:360*400);
k = r100.ann_s < numel(r100.ecg_mv); r100.ann_s = r100.ann_s(k); r100.ann_cls = r100.ann_cls(k); r100.ann_code = r100.ann_code(k);
r119 = mitdb_read('data', 119); r119.ecg_mv = r119.ecg_mv(1:360*400);
adc = afe_twin(r119.ecg_mv, afe_default()); S = stream_prepare(adc, cfg, struct());
R = [S.raw.R]'; qw = [S.raw.qrs_w]'; r119.ann_s = R(~[S.raw.drop]); r119.ann_cls = 1 + 2 * (qw(~[S.raw.drop]) >= 22);
r119.ann_code = zeros(size(r119.ann_s));
o = struct('records_override', {{r100, r119}}, 'ds1_ids', [100 119], 'ds2_ids', [100 119], 'val_ids', 119, ...
           'epochs', 6, 'n_iter', 2, 'export', false, 'results_dir', 'results_smoke', 'tag', 'smoke');
Rr = train_all(o);
