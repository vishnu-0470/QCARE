function e = qat_exp_for_values(v)
% QAT_EXP_FOR_VALUES  Power-of-two exponent for an 8-bit unsigned activation tensor.
%   Chosen so that the 99.9th percentile of the calibration values fits in
%   127 * 2^e.  Values are non-negative (post-ReLU).
s = sort(v(:));
p = s(max(1, ceil(0.999 * numel(s))));
if p <= 0
    e = -7;
else
    e = ceil(log2(p / 127));
end
e = max(min(e, 8), -20);
end
