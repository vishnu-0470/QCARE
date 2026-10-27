function [tau, tab] = calibrate_tau(T, Xv, Yv, max_err, max_leak)
% CALIBRATE_TAU  Smallest early-exit purity threshold such that, on held-out
%   beats, (a) the early-exit error rate <= max_err and (b) true S/V beats
%   wrongly early-exited as N <= max_leak of all held-out S/V beats.
if nargin < 4, max_err = 0.003; end
if nargin < 5, max_leak = 0.005; end
lv = tree_route(T, Xv); pur = T.pur(lv + 1); lc = T.cls(lv + 1);
nsv = max(1, sum(Yv > 0)); tau = 256; tab = zeros(0, 4);
for t = 128:255
    e = pur >= t; ne = sum(e);
    if ne == 0, continue; end
    err = mean(lc(e) ~= Yv(e)); leak = sum(e & lc == 0 & Yv > 0) / nsv;
    tab(end + 1, :) = [t, ne / numel(Yv), err, leak]; %#ok<AGROW>
    if err <= max_err && leak <= max_leak && tau == 256, tau = t; end
end
end
