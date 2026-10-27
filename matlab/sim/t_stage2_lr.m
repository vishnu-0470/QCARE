1;
script_dir = fileparts(mfilename('fullpath')); root = fileparts(fileparts(script_dir));
cd(root); addpath(genpath(fullfile(root,'matlab')));
r = mitdb_read('data', 100); ann = r.ann_s + 1;
maps = zeros(10, 1, 60, 60);
for k = 1:10, R = ann(k + 2); maps(k,1,:,:) = cwt_stage1(r.ecg_mv(R-140:R+219)); end
Y = mod((0:9)', 5) + 1;
for lr = [0.02 0.1]
  for ep_max = [30 80]
    Ptr = irn_net_init(2); acc = 0; L = 0;
    for ep = 1:ep_max
        tot = 0; corr = 0;
        for s = 1:10
            X = reshape(maps(s,1,:,:), 1, 60, 60);
            [lg, c] = irn_net_forward(X, Ptr);
            [l, dz] = nn_softmax_ce(lg, Y(s));
            tot = tot + l; [~, pk] = max(lg); corr = corr + (pk == Y(s));
            G = irn_net_backward(dz, c, Ptr);
            Ptr = sgd_step_(Ptr, G, lr);
        end
        L = tot/10; acc = corr/10;
    end
    fprintf('lr=%.2f epochs=%d  final loss %.4f  train acc %.0f%%\n', lr, ep_max, L, 100*acc);
  end
end

function P = sgd_step_(P, G, lr)
for a = {'ds1','irn1','irn2','ds2','irn3','irn4','fc'}
    for f = fieldnames(P.(a{1})).'
        P.(a{1}).(f{1}) = P.(a{1}).(f{1}) - lr * G.(a{1}).(f{1});
    end
end
end
