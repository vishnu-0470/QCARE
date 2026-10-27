1;
script_dir = fileparts(mfilename('fullpath')); root = fileparts(fileparts(script_dir));
cd(root); addpath(genpath(fullfile(root,'matlab')));
randn('state', 11); rand('state', 4);
Xr = randn(10,1,60,60); Xr = max(min(Xr,1),-1);
qr = min(max(floor(Xr*128+0.5),-128),127); Yr = mod((0:9)',5);
P0 = irn_net_init(1); Er = qat_calibrate(P0, qr);
[Pq, h] = qat_train(P0, Er, qr, Yr, struct('epochs', 80, 'lr', 0.01, 'batch', 5, 'momentum', 0.8));
fprintf('paper-style SGD (mom 0.8, lr 0.01, batch 5), 80 ep: loss %s | acc %s\n', ...
  sprintf('%.3f ', h([1 10 20 40 60 80],1)), sprintf('%.0f%% ', 100*h([1 10 20 40 60 80],2)));
