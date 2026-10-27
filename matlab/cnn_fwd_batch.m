function [logits, c] = cnn_fwd_batch(U, FQ, P, quant)
% CNN_FWD_BATCH  Vectorised forward pass of the stage-2 network.
%   quant=true  : integer-exact QAT forward (identical to cnn_forward_int):
%                 4-bit weights, rounded biases, floor-shift requantisation.
%   quant=false : smooth float forward, used only for the gradient check.
L = 120; B = size(U, 1);
if quant, q = @(w) min(max(round(w), -8), 7); fl = @floor; rb = @round;
else, q = @(w) w; fl = @(x) x; rb = @(x) x; end
W1 = q(P.W1f); W2 = q(P.W2f); Wf1 = q(P.Wf1f); Wf2 = q(P.Wf2f);
b1 = rb(P.b1); b2 = rb(P.b2); bf1 = rb(P.bf1); bf2 = rb(P.bf2);
up = [zeros(B, 1), U, zeros(B, 1)];
a1 = zeros(B, L, 8); y1 = a1;
for oc = 1:8
    a = b1(oc) + W1(oc,1) * up(:, 1:L) + W1(oc,2) * up(:, 2:L+1) + W1(oc,3) * up(:, 3:L+2);
    a1(:,:,oc) = a; y1(:,:,oc) = min(max(fl(a / 2^P.s1), 0), 127);
end
yp = zeros(B, L + 2, 4); yp(:, 2:L+1, :) = y1(:, :, 1:4);
a2 = zeros(B, L, 4); z = a2; out = y1;
for oc = 1:4
    a = b2(oc) * ones(B, L);
    for ic = 1:4, for k = 1:3, a = a + W2(oc,ic,k) * yp(:, k:k+L-1, ic); end, end
    a2(:,:,oc) = a; z(:,:,oc) = min(max(fl(a / 2^P.s2), 0), 127);
    out(:,:,oc) = min(127, z(:,:,oc) + y1(:,:,oc));
end
vec = zeros(B, 40);
for ch = 1:8
    o = out(:,:,ch);
    vec(:, ch)      = max(o, [], 2);
    vec(:, 8 + ch)  = fl(sum(o(:, 23:38), 2) / 16);
    vec(:, 16 + ch) = max(o(:, 39:54), [], 2);
    vec(:, 24 + ch) = fl(sum(o(:, 61:92), 2) / 32);
end
vec(:, 33:40) = FQ;
a3 = bf1(:).' + vec * Wf1.';
h = min(max(fl(a3 / 2^P.s3), 0), 127);
logits = bf2(:).' + h * Wf2.';
c = struct('up', {up}, 'a1', {a1}, 'y1', {y1}, 'yp', {yp}, 'a2', {a2}, 'z', {z}, 'out', {out}, ...
           'vec', {vec}, 'a3', {a3}, 'h', {h}, 'W1', {W1}, 'W2', {W2}, 'Wf1', {Wf1}, 'Wf2', {Wf2});
end
