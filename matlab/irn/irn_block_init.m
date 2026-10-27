function B = irn_block_init(C)
% IRN_BLOCK_INIT  Parameters of one Inception-ResNeXt block with C channels (C even).
%   1x1 conv C->C, 3x3 conv C->C, 3x3 conv on half the channels (C/2 -> C/2).
h = C / 2;
B.W1 = he_init([C C 1 1], C);       B.b1 = zeros(C, 1);
B.W2 = he_init([C C 3 3], 9 * C);   B.b2 = zeros(C, 1);
B.W3 = he_init([h h 3 3], 9 * h);   B.b3 = zeros(h, 1);
end

function W = he_init(sz, fan_in)
W = randn(sz) * sqrt(2 / fan_in);
end
