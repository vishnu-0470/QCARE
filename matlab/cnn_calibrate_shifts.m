function c = cnn_calibrate_shifts(U, FQ, c, cfg)
% CNN_CALIBRATE_SHIFTS  Pick s1,s2,s3 (power-of-two requantisation) so that the
%   95th percentile of each ReLU output is about 64 (of 127) on calibration data.
L = cfg.cnn.L; n = min(size(U, 1), 300); idx = round(linspace(1, size(U, 1), n));
a1 = [];
for i = idx
    up = [0, U(i,:), 0];
    for oc = 1:8
        a1 = [a1, c.w1(oc,1)*up(1:L) + c.w1(oc,2)*up(2:L+1) + c.w1(oc,3)*up(3:L+2) + c.b1(oc)]; %#ok<AGROW>
    end
end
c.s1 = shift_for(a1);
a2 = [];
for i = idx
    up = [0, U(i,:), 0]; y1 = zeros(8, L);
    for oc = 1:8
        acc = c.w1(oc,1)*up(1:L) + c.w1(oc,2)*up(2:L+1) + c.w1(oc,3)*up(3:L+2) + c.b1(oc);
        y1(oc,:) = min(max(floor(acc / 2^c.s1), 0), 127);
    end
    yp = [zeros(8,1), y1, zeros(8,1)];
    for oc = 1:4
        acc = c.b2(oc) * ones(1, L);
        for ic = 1:4, for k = 1:3, acc = acc + c.w2(oc,ic,k) * yp(ic, k:k+L-1); end, end
        a2 = [a2, acc]; %#ok<AGROW>
    end
end
c.s2 = shift_for(a2);
c0 = c; c0.s3 = 0; vs = zeros(numel(idx), 40);
for q = 1:numel(idx), [~, ~, vs(q,:)] = cnn_forward_int(U(idx(q),:), FQ(idx(q),:), c0, cfg); end
a3 = [];
for j = 1:cfg.cnn.H, a3 = [a3; vs * c.wf1(j,:).' + c.bf1(j)]; end %#ok<AGROW>
c.s3 = shift_for(a3);
end

function s = shift_for(a)
a = sort(max(a(:), 0)); p = a(max(1, round(0.95 * numel(a))));
s = max(0, ceil(log2(max(p, 1) / 64)));
end
