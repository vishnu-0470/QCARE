script_dir = fileparts(mfilename('fullpath')); root = fileparts(fileparts(script_dir));
cd(root); addpath(genpath(fullfile(root,'matlab')));
cfg = hw_cfg();
r = mitdb_read('data', 100);
N = 360*300;
adc = afe_twin(r.ecg_mv(1:N), struct());
% pass 1: features only, to calibrate shifts of the random CNN
[b0, r0] = run_stream_golden(adc, cfg, [], struct('classify', false));
U = cell2mat(cellfun(@(q) q.u, r0(~[b0.drop]), 'UniformOutput', false));
FQ = zeros(size(U,1), 8); k = 0;
for i = 1:numel(b0), if ~b0(i).drop, k = k + 1; fv = b0(i).fv; for j = 1:8, if j==7, FQ(k,j)=fv(7)*64; else FQ(k,j)=min(max(floor((fv(j)-256)/4),-128),127); end, end, end, end
model = make_random_model(7, cfg, U, FQ);
fprintf('calibrated shifts: s1=%d s2=%d s3=%d\n', model.cnn.s1, model.cnn.s2, model.cnn.s3);
tic; [b, raws, ev] = run_stream_golden(adc, cfg, model); fprintf('golden run: %d events, %d beats processed in %.1fs\n', numel(ev), numel(b), toc);
cls = [b.cls]; fprintf('classes: N=%d S=%d V=%d drop=%d | early-exit=%.1f%%\n', sum(cls==0), sum(cls==1), sum(cls==2), sum(cls==3), 100*mean([b.early]));
fv = reshape([b.fv], 10, []).'; v = fv(~[b.drop],:);
fprintf('feature medians (rel): '); fprintf('%d ', round(median(v))); fprintf('\n');
q = cellfun(@(x) x.qrs_w, raws(~[b.drop])); fprintf('qrs_w median=%d  [p5 p95]=[%d %d]\n', median(q), min(q), max(q));
R = [b.R]; ann = r.ann_s(r.ann_s < N); d = zeros(size(R)); for i=1:numel(R), [~,j]=min(abs(ann-R(i))); d(i)=R(i)-ann(j); end
fprintf('refined R minus annotation: median=%d  min=%d  max=%d samples\n', median(d), min(d), max(d));
