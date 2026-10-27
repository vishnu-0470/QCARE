script_dir = fileparts(mfilename('fullpath')); root = fileparts(fileparts(script_dir));
cd(root); addpath(genpath(fullfile(root,'matlab')));
cfg = hw_cfg();
r = mitdb_read('data', 100); r2 = mitdb_read('data', 119);
adcA = afe_twin(r.ecg_mv(1:360*120), struct());
adcB = afe_twin(r2.ecg_mv(1:360*120), struct());
out_file = fullfile(root,'experiments','matlab_beats_dump.txt');
fid = fopen(out_file, 'w');
for rec = 1:2
    if rec == 1, adc = adcA; else, adc = adcB; end
    [b, raws] = run_stream_golden(adc, cfg, [], struct('classify', false));
    idx = [20 21 40 41 60 61 80 81]; if rec == 2, idx = [10 30 45 60 75 90 105 120]; end
    for q = idx
        if q > numel(b) || b(q).drop, continue; end
        R = b(q).R; w = adc(R-139:R+220); rw = raws{q};
        fprintf(fid, '%d %d %d %d %d %d %d %d %d %d ', rec, R, rw.onset, rw.offset, rw.base, rw.qrs_w, rw.p_amp, rw.t_amp, rw.tpol, rw.pr);
        fprintf(fid, '%d ', w); fprintf(fid, '\n');
    end
end
fprintf('Beat dump: %s\n', out_file);
fclose(fid);
