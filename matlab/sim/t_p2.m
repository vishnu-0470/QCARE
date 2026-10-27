script_dir = fileparts(mfilename('fullpath')); root = fileparts(fileparts(script_dir));
cd(root); addpath(genpath(fullfile(root,'matlab')));
r = mitdb_read('data', 100);
u = unique(r.ann_code); 
for k=1:numel(u), fprintf('code %d: %d\n', u(k), sum(r.ann_code==u(k))); end
i = find(r.ann_cls ~= 1 & r.ann_cls ~= 2); disp([r.ann_s(i) r.ann_code(i)]);
