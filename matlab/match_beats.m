function [lbl, ann_det] = match_beats(R, ann_s, ann_cls, tol)
% MATCH_BEATS  One-to-one matching of detections to annotations within tol
%   samples (EC57 uses 150 ms = 54 samples).  lbl(j) = AAMI class of the
%   annotation matched to detection j (0 = false detection); ann_det(i) =
%   matched detection of annotation i (0 = missed).
R = R(:); lbl = zeros(numel(R), 1); ann_det = zeros(numel(ann_s), 1); used = false(numel(R), 1);
for i = 1:numel(ann_s)
    d = abs(R - ann_s(i)); d(used) = Inf; [m, j] = min(d);
    if ~isempty(m) && m <= tol, used(j) = true; lbl(j) = ann_cls(i); ann_det(i) = j; end
end
end
