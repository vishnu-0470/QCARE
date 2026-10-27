function cls = aami_label(code)
% AAMI_LABEL  Map WFDB annotation codes to AAMI EC57 classes.
%   1=N 2=S 3=V 4=F 5=Q, 0 = not a beat annotation (ignored).
%   N: N(1) L(2) R(3) j(11) e(34)       S: a(4) J(7) A(8) S(9) n(35)
%   V: V(5) E(10) r(41)                 F: F(6)
%   Q: /(12) Q(13) f(38)
lut = zeros(1, 64);                       % lut(code+1) = class
lut([1 2 3 11 34] + 1) = 1;
lut([4 7 8 9 35]  + 1) = 2;
lut([5 10 41]     + 1) = 3;
lut(6  + 1)            = 4;
lut([12 13 38]    + 1) = 5;
cls = zeros(size(code));
ok = code >= 0 & code < 64;
cls(ok) = lut(code(ok) + 1);
end
