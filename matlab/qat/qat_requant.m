function q = qat_requant(acc, k)
% QAT_REQUANT  Re-scale an integer accumulator by a power of two, with rounding.
%   acc : integer-valued numeric array (accumulator, real value = acc * 2^e_acc)
%   k   : e_dst - e_acc (integer).  Result q has real value q * 2^e_dst.
%   Rounding is round-half-up: floor(x + 0.5), identical to a hardware
%   (x + 2^(k-1)) >> k with arithmetic shift.
% MATLAB integer arrays do not support division by a nonscalar double array.
% The QAT forward path uses exact integer-valued doubles, so normalize inputs
% here before power-of-two scaling; all accumulator values remain exact.
acc = double(acc);
k = double(k);
if k > 0
    q = floor(acc / 2^k + 0.5);
elseif k < 0
    q = acc * 2^(-k);
else
    q = acc;
end
end
