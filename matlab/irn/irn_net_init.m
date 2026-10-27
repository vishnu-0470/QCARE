function P = irn_net_init(seed)
% IRN_NET_INIT  Parameters of the two-stage 2D Inception-ResNeXt network
%   of Cao et al. Fig. 1 / Fig. 10(a):
%     input 1x60x60 (CWT map)
%     DS1  conv3x3 stride 3, 1->8    -> 8x20x20  (ReLU)
%     IRN1, IRN2 (C=8)               -> 8x20x20
%     maxpool 2x2                    -> 8x10x10
%     DS2  conv3x3 stride 2, 8->16   -> 16x4x4   (ReLU)
%     IRN3, IRN4 (C=16)              -> 16x4x4
%     global average pool            -> 16
%     FC 16 -> 5 (N, S, V, F, Q)
if nargin < 1, seed = 1; end
if exist('OCTAVE_VERSION', 'builtin')
    randn('state', seed);
else
    rng(seed);
end
P.ds1.W = randn(8, 1, 3, 3) * sqrt(2 / 9);   P.ds1.b = zeros(8, 1);
P.irn1  = irn_block_init(8);
P.irn2  = irn_block_init(8);
P.ds2.W = randn(16, 8, 3, 3) * sqrt(2 / 72); P.ds2.b = zeros(16, 1);
P.irn3  = irn_block_init(16);
P.irn4  = irn_block_init(16);
P.fc.W  = randn(5, 16) * sqrt(2 / 16);       P.fc.b = zeros(5, 1);
end
