% MAKE_VECTORS  Build RTL regression vectors for one record segment.
%   Run from the project root with variables preset, e.g.
%   octave-cli --eval "REC=119;T0=0;DUR=120;TAG='rec119';SEED=11;GAIN=38;NOISE=5;ART=1;TAU=150;source('matlab/sim/make_vectors.m')"
%   ART=1 injects a saturating offset burst and a flat-line (lead-off) burst.
if ~exist('REC', 'var'),  REC = 100; end
if ~exist('T0', 'var'),   T0 = 0; end
if ~exist('DUR', 'var'),  DUR = 90; end
if ~exist('TAG', 'var'),  TAG = 'rec100'; end
if ~exist('SEED', 'var'), SEED = 7; end
if ~exist('GAIN', 'var'), GAIN = 38; end
if ~exist('NOISE', 'var'), NOISE = 0; end
if ~exist('ART', 'var'),  ART = 0; end
if ~exist('TAU', 'var'),  TAU = 150; end
script_dir = fileparts(mfilename('fullpath')); root = fileparts(fileparts(script_dir));
cd(root); addpath(genpath(fullfile(root,'matlab')));
cfg = hw_cfg();
r = mitdb_read('data', REC);
i0 = round(T0 * 360) + 1; i1 = min(numel(r.ecg_mv), i0 + round(DUR * 360) - 1);
x = r.ecg_mv(i0:i1);
if ART
    n = numel(x);
    a = round(0.30 * n); x(a:a + 900) = x(a:a + 900) + 40;          % overload burst (40 mV)
    b = round(0.60 * n); x(b:b + 1100) = x(b);                        % lead-off flat line
    c = round(0.80 * n); x(c:c + 400) = x(c:c + 400) + 3 * sin((0:400)' * 0.9); % interference burst
end
p = struct('gain_dB', GAIN, 'noise_uVrms', NOISE, 'seed', SEED, 'rec_slow', 3);
adc = afe_twin(x, p);
[b0, r0] = run_stream_golden(adc, cfg, [], struct('classify', false));
keep = find(~[b0.drop]);
U = zeros(numel(keep), 120); FQ = zeros(numel(keep), 8);
for q = 1:numel(keep)
    U(q, :) = r0{keep(q)}.u; fv = b0(keep(q)).fv;
    for j = 1:8
        if j == 7, FQ(q, j) = fv(7) * 64; else, FQ(q, j) = min(max(floor((fv(j) - 256) / 4), -128), 127); end
    end
end
model = make_random_model(SEED, cfg, U, FQ);
model.tau = TAU;
export_rtl(model, cfg);
export_golden(adc, cfg, model, TAG);
fprintf('%s: %d samples, shifts s1=%d s2=%d s3=%d tau=%d\n', TAG, numel(adc), model.cnn.s1, model.cnn.s2, model.cnn.s3, model.tau);
fid = fopen(['xilinx/sim/' TAG '_ns.txt'], 'w'); fprintf(fid, '%d\n', numel(adc)); fclose(fid);
