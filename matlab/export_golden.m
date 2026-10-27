function export_golden(adc, cfg, model, tag, out_dir)
% EXPORT_GOLDEN  Write the ADC stream and the golden per-beat log used by
%   sim/tb_accel.v for bit-exact RTL regression.
if nargin < 5, out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'sim'); end
f = fopen(fullfile(out_dir, [tag '_stream.mem']), 'w');
fprintf(f, '%03x\n', adc(:)); fclose(f);
[b, raws] = run_stream_golden(adc, cfg, model);
f = fopen(fullfile(out_dir, [tag '_golden.txt']), 'w');
for i = 1:numel(b)
    o = b(i); rw = raws{i};
    if o.drop
        fprintf(f, 'D %d %d %d\n', o.c, o.R, rw.base);
    else
        lg = o.logits; shp = rw.sh; if o.early, lg = [0 0 0]; shp = 0; end
        fprintf(f, 'B %d %d %d %d %d %d %d %d %d %d %d %d %d', o.c, o.R, o.cls, o.early, rw.base, ...
                rw.qrs_w, rw.p_amp, rw.t_amp, rw.ramp, rw.tpol, rw.rt, rw.pr, shp);
        fprintf(f, ' %d', o.fv); fprintf(f, ' %d %d %d %d %d\n', o.leaf_cls, o.leaf_pur, lg);
    end
end
fclose(f);
fprintf('golden: %d beats (%d dropped) written for %s\n', numel(b), sum([b.drop]), tag);
end
