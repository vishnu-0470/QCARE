function [logit, cls, C] = qnn_int_forward(qin, Qm)
% QNN_INT_FORWARD  Integer-exact forward pass of the 4W/8A network.
%   qin : [1 60 60] integers in [-128,127]   (real input = qin * 2^-7)
%   Returns logit [5x1] (integers), cls (0..4 = N,S,V,F,Q), and C, a struct of
%   integer tensors used for the STE backward pass and for hardware checks.
%   Every requantisation is qat_requant(acc, e_dst - e_acc) followed by a
%   clamp to [0,127] (all activations are post-ReLU, 8-bit unsigned range).
E = Qm.E; C = struct();
C.qin = qin;

acc = nn_conv2d_fwd(qin, Qm.ds1.W, Qm.ds1.b, 3, 0);
C.acc_ds1 = acc;
D1 = clamp8(qat_requant(acc, E.ds1 - Qm.ds1.eA));

[U1, Cb1] = block_int(D1, Qm.irn1);
[U2, Cb2] = block_int(U1, Qm.irn2);
[P2, pc] = nn_maxpool2_fwd(U2);
C.pc = pc; C.psz = size(U2); C.P2 = P2;

acc = nn_conv2d_fwd(P2, Qm.ds2.W, Qm.ds2.b, 2, 0);
C.acc_ds2 = acc;
D2 = clamp8(qat_requant(acc, E.ds2 - Qm.ds2.eA));

[U3, Cb3] = block_int(D2, Qm.irn3);
[U4, Cb4] = block_int(U3, Qm.irn4);
C.v4sz = size(U4);
HW = size(U4, 2) * size(U4, 3);
gsum = reshape(sum(reshape(U4, size(U4, 1), HW), 2), size(U4, 1), 1);
g = floor(gsum / HW + 0.5);                       % integer mean, round-half-up
g = clamp8(g);
C.g = g; C.D1 = D1; C.D2 = D2; C.U1 = U1; C.U2 = U2; C.U3 = U3; C.U4 = U4;
C.cb = {Cb1, Cb2, Cb3, Cb4};

logit = Qm.fc.W * g + Qm.fc.b;
C.logit = logit;
[~, ix] = max(logit);                             % first maximum wins
cls = ix - 1;
C.cls = cls;
end

function [U, Cb] = block_int(x, B)
% Integer Inception-ResNeXt block; mirrors irn_block_fwd exactly.
C = size(x, 1); h = C / 2;
acc1 = nn_conv2d_fwd(x, B.W1, B.b1, 1, 0);
A = clamp8(qat_requant(acc1, B.eb - B.eA1));
acc2 = nn_conv2d_fwd(A, B.W2, B.b2, 1, 1);
Bm = clamp8(qat_requant(acc2, B.eb - B.eA2));
B1 = Bm(1:h, :, :);
acc3 = nn_conv2d_fwd(B1, B.W3, B.b3, 1, 1);
B1p = clamp8(qat_requant(acc3, B.eb - B.eA3));
Z = cat(1, B1p, Bm(h + 1:C, :, :)) + A;          % same scale: plain integer add
U = clamp8(Z);
Cb = struct('x', x, 'acc1', acc1, 'A', A, 'acc2', acc2, 'B1', B1, 'acc3', acc3, 'Z', Z);
end

function y = clamp8(x)
y = min(max(x, 0), 127);
end
