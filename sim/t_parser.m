addpath('../matlab');
r = mitdb_read('../data', 100);
fprintf('record 100: fs=%d lead=%s samples=%d beats=%d\n', r.fs, r.lead_name, numel(r.ecg_mv), numel(r.ann_s));
for c = 1:5, fprintf('  class %d: %d\n', c, sum(r.ann_cls == c)); end
fprintf('  ecg_mv min/max/mean = %.3f / %.3f / %.3f   first ann = %d (code %d)\n', min(r.ecg_mv), max(r.ecg_mv), mean(r.ecg_mv), r.ann_s(1), r.ann_code(1));
r2 = mitdb_read('../data', 119);
fprintf('record 119 (waveform only): samples=%d  ecg_mv min/max = %.2f / %.2f  hea?=%d\n', numel(r2.ecg_mv), min(r2.ecg_mv), max(r2.ecg_mv), exist('../data/119.hea','file')==2);
