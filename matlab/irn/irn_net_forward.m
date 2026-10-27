function [logits, cache] = irn_net_forward(X, P)
% IRN_NET_FORWARD  Forward pass of the stage-2 network on one map X [1 60 60].
cache.X = X;
[z1, ~] = nn_conv2d_fwd(X, P.ds1.W, P.ds1.b, 3, 0);
cache.z1 = z1;
h1 = max(z1, 0);
[u1, c1] = irn_block_fwd(h1, P.irn1);
[u2, c2] = irn_block_fwd(u1, P.irn2);
cache.c1 = c1; cache.c2 = c2;
[p, pc] = nn_maxpool2_fwd(u2);
cache.pc = pc; cache.psz = size(u2);
[z2, ~] = nn_conv2d_fwd(p, P.ds2.W, P.ds2.b, 2, 0);
cache.z2 = z2;
h2 = max(z2, 0);
[v3, c3] = irn_block_fwd(h2, P.irn3);
[v4, c4] = irn_block_fwd(v3, P.irn4);
cache.c3 = c3; cache.c4 = c4; cache.v4sz = size(v4);
g = nn_gap_fwd(v4);
cache.g = g;
logits = nn_fc_fwd(g, P.fc.W, P.fc.b);
cache.h1 = h1; cache.h2 = h2; cache.p = p;
end
