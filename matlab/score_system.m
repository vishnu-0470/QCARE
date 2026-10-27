function M = score_system(Plist, Dlist)
% SCORE_SYSTEM  AAMI/EC57-style scoring over annotated beats.
%   Rows of M.conf = true AAMI class (N,S,V,F,Q); columns = outcome
%   (N,S,V,dropped-by-gate,missed).  Dropped and missed beats count as
%   false negatives; false detections classified as N/S/V count as false
%   positives (M.fp).  Per-class sensitivity (Se) and positive predictivity.
conf = zeros(5, 5); fp = zeros(1, 3); ndet = 0; ntp = 0; nann = 0; nproc = 0; nearly = 0; ndrop = 0;
for r = 1:numel(Plist)
    P = Plist{r}; D = Dlist{r}; nann = nann + P.n_ann; ndet = ndet + numel(D.cls); ntp = ntp + sum(P.ann_det > 0);
    for i = 1:P.n_ann
        j = P.ann_det(i); out = 5;
        if j > 0, c = D.cls(j); if c < 3, out = c + 1; else, out = 4; end, end
        conf(P.ann_cls(i), out) = conf(P.ann_cls(i), out) + 1;
    end
    um = find(P.lbl == 0 & D.cls < 3);
    for q = 1:numel(um), fp(D.cls(um(q)) + 1) = fp(D.cls(um(q)) + 1) + 1; end
    nproc = nproc + numel(D.cls); ndrop = ndrop + sum(D.cls == 3); nearly = nearly + sum(D.early & D.cls < 3);
end
M.conf = conf; M.fp = fp;
names = {'N', 'S', 'V'};
M.Se = zeros(1,3); M.PPV = zeros(1,3); M.F1 = zeros(1,3);
for c = 1:3
    tp = conf(c, c); fn = sum(conf(c, :)) - tp; fpc = sum(conf(:, c)) - tp + fp(c);
    M.Se(c) = tp / max(1, tp + fn); M.PPV(c) = tp / max(1, tp + fpc);
    M.F1(c) = 2 * M.Se(c) * M.PPV(c) / max(1e-9, M.Se(c) + M.PPV(c));
end
M.acc = (conf(1,1) + conf(2,2) + conf(3,3)) / max(1, sum(conf(:)));
M.det_se = ntp / max(1, nann); M.det_ppv = ntp / max(1, ndet);
M.early_frac = nearly / max(1, nproc - ndrop); M.drop_frac = ndrop / max(1, nproc);
M.n_ann = nann; M.names = names;
end
