function [out, cache] = irn_block_fwd(x, B)
% IRN_BLOCK_FWD  Inception-ResNeXt block (Cao et al., Fig. 4b), C in -> C out.
%   a1 = conv1x1(x);            A  = relu(a1)                 (reduce + residual source)
%   a2 = conv3x3(A);            B  = relu(a2)
%   split B along channels into B1 (first C/2) and B2 (second C/2)
%   a3 = conv3x3(B1);           B1p = relu(a3)                (5x5 effective RF branch)
%   Z  = concat(B1p, B2) + A                                  (residual add)
%   out = relu(Z)
%   The ReLU after the residual add is an assumption (the text does not state it).
C = size(x, 1);
h = C / 2;
[a1, ~] = nn_conv2d_fwd(x, B.W1, B.b1, 1, 0);
A = max(a1, 0);
[a2, ~] = nn_conv2d_fwd(A, B.W2, B.b2, 1, 1);
Bm = max(a2, 0);
B1 = Bm(1:h, :, :);
B2 = Bm(h + 1:C, :, :);
[a3, ~] = nn_conv2d_fwd(B1, B.W3, B.b3, 1, 1);
B1p = max(a3, 0);
Z = cat(1, B1p, B2) + A;
out = max(Z, 0);
cache.x = x; cache.a1 = a1; cache.A = A; cache.a2 = a2;
cache.B1 = B1; cache.a3 = a3; cache.Z = Z;
end
