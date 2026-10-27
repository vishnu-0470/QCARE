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
project_root = fileparts(script_dir);
cd(project_root);
addpath(genpath(fullfile(project_root, 'matlab')));

source_csv = fullfile(project_root, '2.csv');
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
if t_adc(end) < t_unique(end) - 1e-10
    t_adc(end+1, 1) = t_unique(end);
end
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

vec_dir = fullfile(project_root, 'rtl', 'qnn', 'vec');
if ~exist(vec_dir, 'dir'), mkdir(vec_dir); end
vec_path = fullfile(vec_dir, 'afe_qnn_vec_in.mem');
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
