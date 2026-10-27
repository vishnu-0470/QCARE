1;
script_dir = fileparts(mfilename('fullpath')); root = fileparts(fileparts(script_dir));
cd(root); addpath(genpath(fullfile(root,'matlab')));
if exist('OCTAVE_VERSION', 'builtin'), randn('state', 1); rand('state', 1); else, rng(1); end
r119 = mitdb_read('data', 119); r100 = mitdb_read('data', 100); ann = r100.ann_s + 1;
P = irn_net_init(1);
cal = zeros(6,1,60,60);
for k = 1:6, cal(k,1,:,:) = floor(cwt_stage1(r119.ecg_mv(1000+(k-1)*360:1000+k*360-1))*128+0.5); end
cal = min(max(cal,-128),127);
E = qat_calibrate(P, cal); Qm = qat_build(P, E);
V = zeros(8,1,60,60);
for k = 1:4                                   % 4 real DS2 beats (record 100)
    R = ann(k+2); V(k,1,:,:) = min(max(floor(cwt_stage1(r100.ecg_mv(R-140:R+219))*128+0.5),-128),127);
end
Xr = max(min(randn(4,1,60,60),1),-1);         % 4 random maps
V(5:8,1,:,:) = min(max(floor(Xr*128+0.5),-128),127);
out_dir = fullfile(root,'experiments','matlab_stage4_export');
info = qnn_export(Qm, V, out_dir);
fprintf('exported: %d ops, activation RAM %d bytes, weights %d, biases %d, vectors %d\n', ...
    info.nops, info.ram_used, info.nw, info.nb, info.nvec);
fprintf('output: %s\n', out_dir);
