function R = refine_R(adc, c, cfg)
% REFINE_R  Locate the R peak from the coarse detector event c: the sample
%   with the largest |x - x[c-56]| in [c-40, c-2] (first maximum wins).
e = cfg.eng; ref0 = adc(c - e.ref_back + 1); best = 0; R = c - e.srch_lo;
for i = c - e.srch_lo : c - e.srch_hi
    d = abs(adc(i + 1) - ref0);
    if d > best, best = d; R = i; end
end
end
