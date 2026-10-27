1;
script_dir = fileparts(mfilename('fullpath')); root = fileparts(fileparts(script_dir));
cd(root); addpath(genpath(fullfile(root,'matlab')));
randn('state', 11);
Xr = randn(10,1,60,60); Xr = max(min(Xr,1),-1);
qr = min(max(floor(Xr*128+0.5),-128),127);
P0 = irn_net_init(1); E = qat_calibrate(P0, qr); Qm = qat_build(P0, E);
Lf = zeros(10,5); Lq = zeros(10,5);
for n = 1:10
    X = reshape(qr(n,1,:,:),1,60,60)*2^-7;
    Lf(n,:) = irn_net_forward(X, P0).';
    [lg, ~] = qnn_int_forward(reshape(qr(n,1,:,:),1,60,60), Qm);
    Lq(n,:) = (double(lg)*2^(E.blk4+Qm.fc.eW)).';
end
spread = mean(std(Lf));  err = mean(abs(Lf(:)-Lq(:)));
fprintf('float logit spread across inputs: %.4f\n', spread);
fprintf('mean |float - quantised| logit error: %.4f\n', err);
fprintf('error / spread = %.2f  (>1 means quantisation noise hides the input differences)\n', err/spread);
fprintf('calibrated exponents: ds1 %d blk1 %d blk2 %d ds2 %d blk3 %d blk4 %d\n', E.ds1,E.blk1,E.blk2,E.ds2,E.blk3,E.blk4);
