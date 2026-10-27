function G = irn_net_backward(dlogits, cache, P)
% IRN_NET_BACKWARD  Gradients of the cross-entropy loss w.r.t. every parameter.
[dg, G.fc.W, G.fc.b] = nn_fc_bwd(dlogits, cache.g, P.fc.W);
dv4 = nn_gap_bwd(dg, cache.v4sz);
[dv3, G.irn4] = irn_block_bwd(dv4, cache.c4, P.irn4);
[dh2, G.irn3] = irn_block_bwd(dv3, cache.c3, P.irn3);
dz2 = nn_relu_bwd(dh2, cache.z2);
[dp, G.ds2.W, G.ds2.b] = nn_conv2d_bwd(dz2, cache.p, P.ds2.W, 2, 0);
du2 = nn_maxpool2_bwd(dp, cache.pc, cache.psz);
[du1, G.irn2] = irn_block_bwd(du2, cache.c2, P.irn2);
[dh1, G.irn1] = irn_block_bwd(du1, cache.c1, P.irn1);
dz1 = nn_relu_bwd(dh1, cache.z1);
[~, G.ds1.W, G.ds1.b] = nn_conv2d_bwd(dz1, cache.X, P.ds1.W, 3, 0);
end
