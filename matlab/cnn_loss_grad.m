function [loss, G, logits] = cnn_loss_grad(U, FQ, Y, sw, P, quant)
% CNN_LOSS_GRAD  Weighted cross-entropy and its gradient through the QAT
%   forward (straight-through estimator for round/floor; clamps and the
%   residual saturation are respected).  Y in {0,1,2}, sw = sample weights.
L = 120; B = size(U, 1);
[logits, c] = cnn_fwd_batch(U, FQ, P, quant);
Z = logits / 2^P.sl; Z = Z - max(Z, [], 2); E = exp(Z); Pm = E ./ sum(E, 2);
T = zeros(B, 3); T(sub2ind([B 3], (1:B)', Y + 1)) = 1;
loss = -sum(sw .* log(sum(Pm .* T, 2) + 1e-12)) / B;
dZ = (Pm - T) .* sw / B / 2^P.sl;
G.Wf2f = dZ.' * c.h; G.bf2 = sum(dZ, 1).';
dh = dZ * c.Wf2;
if quant, hv = c.a3 / 2^P.s3; m3 = (hv >= 0) & (hv < 128); else, hv = c.a3 / 2^P.s3; m3 = (hv >= 0) & (hv <= 127); end
dacc3 = dh .* m3 / 2^P.s3;
G.Wf1f = dacc3.' * c.vec; G.bf1 = sum(dacc3, 1).';
dvec = dacc3 * c.Wf1;
% ---- pooling backward -> d(out)
dout = zeros(B, L, 8); bi = (1:B)';
for ch = 1:8
    o = c.out(:,:,ch);
    [~, ia] = max(o, [], 2);
    dout(sub2ind([B L 8], bi, ia, ch * ones(B, 1))) = dout(sub2ind([B L 8], bi, ia, ch * ones(B, 1))) + dvec(:, ch);
    dout(:, 23:38, ch) = dout(:, 23:38, ch) + repmat(dvec(:, 8 + ch) / 16, 1, 16);
    [~, iq] = max(o(:, 39:54), [], 2); iq = iq + 38;
    dout(sub2ind([B L 8], bi, iq, ch * ones(B, 1))) = dout(sub2ind([B L 8], bi, iq, ch * ones(B, 1))) + dvec(:, 16 + ch);
    dout(:, 61:92, ch) = dout(:, 61:92, ch) + repmat(dvec(:, 24 + ch) / 32, 1, 32);
end
% ---- conv2 + residual backward
dy1 = zeros(B, L, 8); dy1(:, :, 5:8) = dout(:, :, 5:8);
dypad = zeros(B, L + 2, 4);
G.W2f = zeros(4, 4, 3); G.b2 = zeros(4, 1);
for oc = 1:4
    res = (c.z(:,:,oc) + c.y1(:,:,oc)) <= 127;
    dsum = dout(:,:,oc) .* res;
    dy1(:,:,oc) = dy1(:,:,oc) + dsum;
    zv = c.a2(:,:,oc) / 2^P.s2;
    if quant, mz = (zv >= 0) & (zv < 128); else, mz = (zv >= 0) & (zv <= 127); end
    dacc2 = dsum .* mz / 2^P.s2;
    G.b2(oc) = sum(dacc2(:));
    for ic = 1:4
        for k = 1:3
            G.W2f(oc, ic, k) = sum(sum(dacc2 .* c.yp(:, k:k+L-1, ic)));
            dypad(:, k:k+L-1, ic) = dypad(:, k:k+L-1, ic) + c.W2(oc, ic, k) * dacc2;
        end
    end
end
dy1(:, :, 1:4) = dy1(:, :, 1:4) + dypad(:, 2:L+1, :);
% ---- conv1 backward
G.W1f = zeros(8, 3); G.b1 = zeros(8, 1);
for oc = 1:8
    av = c.a1(:,:,oc) / 2^P.s1;
    if quant, m1 = (av >= 0) & (av < 128); else, m1 = (av >= 0) & (av <= 127); end
    dacc1 = dy1(:,:,oc) .* m1 / 2^P.s1;
    G.b1(oc) = sum(dacc1(:));
    for k = 1:3, G.W1f(oc, k) = sum(sum(dacc1 .* c.up(:, k:k+L-1))); end
end
end
