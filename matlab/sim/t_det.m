script_dir = fileparts(mfilename('fullpath')); root = fileparts(fileparts(script_dir));
cd(root); addpath(genpath(fullfile(root,'matlab')));
cfg = hw_cfg(); prctile_ = @(x,p) subsref(sort(x(:)), struct('type','()','subs',{{max(1, round(p/100*numel(x)))}}));
r = mitdb_read('data', 100);
% ---- ideal chain (no AFE impairments) on the first 10 min of record 100 --
N = 360 * 600;
[adc, info] = afe_twin(r.ecg_mv(1:N), struct('noise_uVrms', 0));
fprintf('twin: gain=%.1f V/V  1 LSB = %.2f uV (input)  clipped fraction=%.4f\n', info.gain_vv, info.lsb_in_uV, info.sat_frac);
tic; [c, dbg] = qrs_detect_golden(adc, cfg); fprintf('detector: %d events in %.1fs\n', numel(c), toc);
ann = r.ann_s(r.ann_s < N);
% lag of c relative to annotation, matched within 150 ms
tol = 54; used = false(size(c)); tp = 0; lag = [];
for i = 1:numel(ann)
    [dmin, j] = min(abs(c - ann(i)));  
    % consider the lag window: the event is expected AFTER the R annotation
    cand = find(c - ann(i) > -20 & c - ann(i) < 60 & ~used);
    if ~isempty(cand), [~, jj] = min(abs(c(cand) - ann(i) - 25)); j = cand(jj); used(j) = true; tp = tp + 1; lag(end+1) = c(j) - ann(i); end
end
fprintf('annotated beats=%d  detected=%d  TP=%d  Se=%.2f%%  +P=%.2f%%\n', numel(ann), numel(c), tp, 100*tp/numel(ann), 100*tp/numel(c));
fprintf('lag c - R_ann: median=%d  min=%d  max=%d  p5=%d  p95=%d samples\n', median(lag), min(lag), max(lag), round(prctile_(lag,5)), round(prctile_(lag,95)));
