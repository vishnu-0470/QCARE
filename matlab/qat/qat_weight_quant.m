function [q, e] = qat_weight_quant(W)
% QAT_WEIGHT_QUANT  4-bit signed weights with a power-of-two scale (Cao et al. Eq. 2-3).
%   kmax = max|W|;  r = floor(log2(kmax))  so that 2^r < kmax < 2^(r+1)   (Eq. 3)
%   LSB exponent e = r + 2 - n,  n = 4   (Eq. 3: LSB = 2^(r_MSB + 2 - n))
%   q = clamp(floor(W / 2^e + 0.5), -8, 7)  ;  real weight = q * 2^e
km = max(abs(W(:)));
if km == 0, r = -10; else, r = floor(log2(km)); end
e = r + 2 - 4;
q = min(max(floor(W / 2^e + 0.5), -8), 7);
end
