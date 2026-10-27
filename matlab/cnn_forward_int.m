function [logits, cls, vec] = cnn_forward_int(u, fq, cnn, cfg)
% CNN_FORWARD_INT  Integer-exact stage-2 network (golden model of the RTL):
%   u[120] int8 -> conv1(1->8,k3)+ReLU>>S1 -> conv2(4->4,k3 on ch0-3)+ReLU>>S2
%   + residual -> regional pooling (32 values) ++ 8 features -> FC(40->16)
%   ReLU>>S3 -> FC(16->3) -> argmax.  Weights are 4-bit signed (-8..7).
%   Returns logits (1x3), class 0/1/2 (first maximum wins) and vec (1x40).
L = cfg.cnn.L;
up = [0, u(:).', 0];
y1 = zeros(8, L);
for oc = 1:8
    acc = cnn.b1(oc) + cnn.w1(oc,1)*up(1:L) + cnn.w1(oc,2)*up(2:L+1) + cnn.w1(oc,3)*up(3:L+2);
    y1(oc,:) = min(max(floor(acc / 2^cnn.s1), 0), 127);
end
yp = [zeros(8,1), y1, zeros(8,1)];
out = y1;
for oc = 1:4
    acc = cnn.b2(oc) * ones(1, L);
    for ic = 1:4
        for k = 1:3
            acc = acc + cnn.w2(oc,ic,k) * yp(ic, k:k+L-1);
        end
    end
    z = min(max(floor(acc / 2^cnn.s2), 0), 127);
    out(oc,:) = min(127, z + y1(oc,:));
end
pp = cfg.cnn.pool_p; pq = cfg.cnn.pool_q; pt = cfg.cnn.pool_t;
vec = zeros(1, 40);
for c = 1:8
    o = out(c,:);
    vec(c)      = max(o);
    vec(8 + c)  = floor(sum(o(pp(1)+1:pp(2)+1)) / 16);
    vec(16 + c) = max(o(pq(1)+1:pq(2)+1));
    vec(24 + c) = floor(sum(o(pt(1)+1:pt(2)+1)) / 32);
end
vec(33:40) = fq(:).';
h = zeros(1, cfg.cnn.H);
for j = 1:cfg.cnn.H
    acc = cnn.bf1(j) + sum(cnn.wf1(j,:) .* vec);
    h(j) = min(max(floor(acc / 2^cnn.s3), 0), 127);
end
logits = zeros(1, 3);
for c = 1:3, logits(c) = cnn.bf2(c) + sum(cnn.wf2(c,:) .* h); end
[~, ix] = max(logits);              % first maximum (strict > in the RTL)
cls = ix - 1;
end
