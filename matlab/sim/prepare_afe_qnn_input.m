% PREPARE_AFE_QNN_INPUT
% Import the Cadence differential output in 2.csv, model the selected ADC,
% convert one 1-second ECG segment to the project's 60x60 int8 CWT map, and
% write the map in the exact row-major format consumed by qnn_engine.
%
% ADC model assumptions for this co-simulation:
%   12-bit offset binary, differential input range -1 V to +1 V, 500 samples/s.
% The existing CWT front end is defined for 360 samples at 360 samples/s.
% With no Cadence R-wave annotations, this script centers the segment on the
% largest absolute deviation from the median as an automatic peak estimate.

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
cd(project_root);
addpath(genpath(fullfile(project_root, 'matlab')));

source_csv = fullfile(project_root, 'data', '2.csv');
if ~exist(source_csv, 'file')
    error('Cadence CSV not found: %s', source_csv);
end
fid = fopen(source_csv, 'rt');
if fid < 0, error('Cannot open %s', source_csv); end
fgetl(fid); % skip CSV header: time (s), differential voltage (V)
raw = textscan(fid, '%f%f', 'Delimiter', ',', 'CollectOutput', true);
fclose(fid);
if isempty(raw) || isempty(raw{1}) || size(raw{1}, 2) < 2
    error('Expected numeric time and differential-voltage columns in 2.csv');
end
t = raw{1}(:, 1);
v = raw{1}(:, 2);
valid = isfinite(t) & isfinite(v);
t = t(valid); v = v(valid);
if numel(t) < 2, error('2.csv must contain at least two valid waveform rows'); end

% Collapse duplicate timestamps (the source export has one rounded duplicate).
[t_unique, ~, group] = unique(t);
v_unique = accumarray(group, v, [], @mean);
[t_unique, order] = sort(t_unique);
v_unique = v_unique(order);
if t_unique(end) - t_unique(1) < 1.0
    error('Need at least 1 second of waveform for a 360-sample segment at 360 Hz');
end

% Resample source waveform to the ADC clock, quantize, then reconstruct the
% sampled voltage so downstream processing sees exactly the ADC-limited data.
adc_bits = 12;
adc_min_v = -1.0;
adc_max_v = 1.0;
adc_fs = 500;
adc_max_code = 2^adc_bits - 1;
t_adc = (t_unique(1):1/adc_fs:t_unique(end)).';
% Keep the ADC clock uniform; never append an off-grid endpoint.
v_adc_ideal = interp1(t_unique, v_unique, t_adc, 'linear');
normalized = (v_adc_ideal - adc_min_v) ./ (adc_max_v - adc_min_v);
adc_code = round(normalized .* adc_max_code);
adc_code = min(max(adc_code, 0), adc_max_code);
v_adc = adc_min_v + double(adc_code) .* (adc_max_v - adc_min_v) ./ adc_max_code;

% The project CWT uses 360 samples per beat at 360 Hz.
model_fs = 360;
t_model = (t_adc(1):1/model_fs:t_adc(end)).';
v_model = interp1(t_adc, v_adc, t_model, 'linear');
segment_length = 360;
left_of_peak = 140;
right_of_peak = segment_length - left_of_peak - 1;
first_valid_peak = left_of_peak + 1;
last_valid_peak = numel(v_model) - right_of_peak;
if last_valid_peak < first_valid_peak
    error('Waveform is too short after resampling to form a 360-sample segment');
end
candidate = first_valid_peak:last_valid_peak;
baseline = median(v_model);
[~, best] = max(abs(v_model(candidate) - baseline));
peak_index = candidate(best);
beat = v_model(peak_index-left_of_peak : peak_index+right_of_peak);

M = cwt_stage1(beat, struct('normalize', true));
if ~isequal(size(M), [60 60])
    error('CWT output must be 60x60; got %dx%d', size(M, 1), size(M, 2));
end
qmap = quantize_map_int8(M);

if ~exist('afe_output_root','var') || isempty(afe_output_root)
    afe_output_root = fullfile(project_root,'experiments','afe_adc_matlab');
end
vec_dir = fullfile(afe_output_root, 'vec');
if ~exist(vec_dir, 'dir'), mkdir(vec_dir); end
vec_path = fullfile(vec_dir, 'qnn_vec_in.mem');
% Preserve reproducible intermediate voltages, ADC codes and maps.
save(fullfile(afe_output_root,'matlab_adapter_values.mat'),'t_adc','adc_code','v_adc','t_model','v_model','peak_index','beat','M','qmap');
% Match qnn_export.m exactly: row-major, signed int8 represented as hex bytes.
flat = reshape(double(qmap.'), 1, []);
fid = fopen(vec_path, 'wt');
if fid < 0, error('Cannot create %s', vec_path); end
for k = 1:numel(flat)
    fprintf(fid, '%02x\n', mod(flat(k), 256));
end
fclose(fid);

fprintf('ADC model: %d-bit, %.3f to %.3f V differential, %d samples/s\n', ...
    adc_bits, adc_min_v, adc_max_v, adc_fs);
fprintf('ADC samples: %d; resampled duration: %.3f s; selected peak at %.6f s\n', ...
    numel(adc_code), t_model(end)-t_model(1), t_model(peak_index));
fprintf('ADC code range: %d..%d; map: 60x60 int8; output: %s\n', ...
    min(adc_code), max(adc_code), vec_path);

% Optional frozen-model reference; default remains a separate output directory.
% Set afe_model_file to a different saved Qm explicitly if needed.
if ~exist('afe_model_file','var') || isempty(afe_model_file)
    afe_model_file=fullfile(project_root,'experiments','ds1_current','frozen_model.mat');
end
if exist(afe_model_file,'file')
    model_state=load(afe_model_file,'Qm');
    [golden_logits,golden_class]=qnn_int_forward(reshape(double(qmap),1,60,60),model_state.Qm);
    fid=fopen(fullfile(vec_dir,'qnn_vec_exp.txt'),'w');
    assert(fid>=0,'Cannot open expected-logit output');
    fprintf(fid,'%d %d %d %d %d %d\n',golden_class,golden_logits(1),golden_logits(2),golden_logits(3),golden_logits(4),golden_logits(5));
    fclose(fid);
    fprintf('Frozen model class %d (no waveform ground truth); expected logits saved.\n',golden_class);
else
    warning('No frozen Qm found; only the input map was exported.');
end
