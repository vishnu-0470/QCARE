function verify_current_export()
% Verify the frozen Python export using the existing MATLAB integer forward.
% Verified on the Windows workstation against ten frozen real DS2 vectors.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));addpath(genpath(fullfile(root,'matlab')));
base=fullfile(root,'experiments','ds1_current');S=load(fullfile(base,'frozen_model.mat'));
G=load(fullfile(base,'export','vec','python_golden.mat'));qmaps=double(G.qmaps);expected=double(G.expected);
assert(size(qmaps,1)==size(expected,1));
for i=1:size(qmaps,1)
    qin=reshape(qmaps(i,1,:,:),1,60,60);
    [lg,cls]=qnn_int_forward(qin,S.Qm);
    assert(cls==expected(i,1),'Class mismatch at vector %d',i);
    assert(isequal(lg(:).',expected(i,2:6)),'Signed logit mismatch at vector %d',i);
end
out=fullfile(base,'matlab_reexport');qnn_export(S.Qm,qmaps,out);
files={'gen/qnn_prog.mem','gen/qnn_w.mem','gen/qnn_b.mem','vec/qnn_vec_in.mem'};
for i=1:numel(files)
    a=read_words(fullfile(base,'export',files{i}));b=read_words(fullfile(out,files{i}));
    assert(isequal(a,b),'ROM/vector order differs in %s',files{i});
end
% Wide instruction file uses 288-bit words: compare its text without parsing
% through double, whose integer precision cannot represent a whole word.
a=lower(regexprep(fileread(fullfile(base,'export','gen','qnn_prog.mem.wide')),'\s',''));
b=lower(regexprep(fileread(fullfile(out,'gen','qnn_prog.mem.wide')),'\s',''));
assert(strcmp(a,b),'Wide instruction words differ');
fprintf('PASS: MATLAB integer logits and export match %d frozen real DS2 vectors\n',size(qmaps,1));
end
function v=read_words(file)
f=fopen(file,'r');assert(f>=0,'Cannot open %s',file);v=fscanf(f,'%x');fclose(f);
end
