function [dx, G] = irn_block_bwd(dout, cache, B)
% IRN_BLOCK_BWD  Backward pass of irn_block_fwd.  Returns dx and gradient struct G.
C = size(cache.x, 1);
h = C / 2;
dZ = nn_relu_bwd(dout, cache.Z);
dB1p = dZ(1:h, :, :);
dB2  = dZ(h + 1:C, :, :);
da3 = nn_relu_bwd(dB1p, cache.a3);
[dB1, G.W3, G.b3] = nn_conv2d_bwd(da3, cache.B1, B.W3, 1, 1);
dBm = cat(1, dB1, dB2);
da2 = nn_relu_bwd(dBm, cache.a2);
[dA, G.W2, G.b2] = nn_conv2d_bwd(da2, cache.A, B.W2, 1, 1);
dA = dA + dZ;                                   % residual branch
da1 = nn_relu_bwd(dA, cache.a1);
[dx, G.W1, G.b1] = nn_conv2d_bwd(da1, cache.x, B.W1, 1, 0);
end
