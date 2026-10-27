1;
script_dir = fileparts(mfilename('fullpath')); root = fileparts(fileparts(script_dir));
cd(root); addpath(genpath(fullfile(root,'matlab')));
randn('state', 11);
Xs = randn(10, 1, 60, 60); Y = mod((0:9)', 5) + 1;   % 10 distinct random maps, cyclic labels
P = irn_net_init(2); lr = 0.005; hist = zeros(1, 60);
for ep = 1:60
    tot = 0; corr = 0;
    for s = 1:10
        [lg, c] = irn_net_forward(reshape(Xs(s,:,:,:),1,60,60), P);
        [l, dz] = nn_softmax_ce(lg, Y(s));
        tot = tot + l; [~, pk] = max(lg); corr = corr + (pk == Y(s));
        P = sgd_step_(P, irn_net_backward(dz, c, P), lr);
    end
    hist(ep) = corr / 10;
    if ep == 1 || ep == 20 || ep == 40 || ep == 60
        fprintf('epoch %2d  loss %.3f  train acc %.0f%%\n', ep, tot/10, 100*corr/10);
    end
end
fprintf('memorisation of 10 random maps: final acc %.0f%%\n', 100*hist(end));

function P = sgd_step_(P, G, lr)
for a = {'ds1','irn1','irn2','ds2','irn3','irn4','fc'}
    for f = fieldnames(P.(a{1})).'
        P.(a{1}).(f{1}) = P.(a{1}).(f{1}) - lr * G.(a{1}).(f{1});
    end
end
end
