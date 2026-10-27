function rec = mitdb_read(data_dir, id)
% MITDB_READ  Read an MIT-BIH record (.hea/.dat format 212, .atr annotations)
%   without any toolbox. Returns the MLII lead in mV plus beat annotations.
%     rec.fs, rec.ecg_mv (column), rec.ann_s (0-based sample), rec.ann_code,
%     rec.ann_cls (AAMI 1..5), rec.lead_name
%   If the .hea is missing, standard MIT-BIH values are assumed (2 signals,
%   format 212, 200 adu/mV, zero 1024, 360 Hz, MLII first) and, if the .atr
%   is also missing, annotations are returned empty (waveform-only use).
base = fullfile(data_dir, sprintf('%d', id));
rec.id = id; rec.fs = 360; nsig = 2; nsamp = Inf;
gain = [200 200]; zero = [1024 1024]; names = {'MLII', 'V1'};
if exist([base '.hea'], 'file')
    txt = fileread([base '.hea']);
    lines = regexp(txt, '[\r\n]+', 'split');
    lines = strtrim(lines);
    lines = lines(~cellfun(@isempty, lines));
    lines = lines(cellfun(@(s) s(1) ~= '#', lines));
    h = strsplit(lines{1});
    nsig = str2double(h{2}); rec.fs = str2double(h{3}); nsamp = str2double(h{4});
    gain = zeros(1, nsig); zero = zeros(1, nsig); names = cell(1, nsig);
    for k = 1:nsig
        f = strsplit(lines{1 + k});
        if ~strcmp(f{2}, '212'), error('mitdb_read: only format 212 supported'); end
        g = regexp(f{3}, '^[0-9.]+', 'match', 'once');
        gain(k) = str2double(g);
        zero(k) = str2double(f{5});
        names{k} = f{end};
    end
end
if nsig ~= 2, error('mitdb_read: expected 2 signals'); end
fid = fopen([base '.dat'], 'r');
if fid < 0, error('mitdb_read: cannot open %s.dat', base); end
raw = fread(fid, inf, 'uint8=>double'); fclose(fid);
nmax = floor(numel(raw) / 3);
if isfinite(nsamp), nmax = min(nmax, nsamp); end
raw = reshape(raw(1:3 * nmax), 3, nmax);
s0 = raw(1, :) + bitshift(bitand(raw(2, :), 15), 8);
s1 = raw(3, :) + bitshift(bitshift(raw(2, :), -4), 8);
s0(s0 >= 2048) = s0(s0 >= 2048) - 4096;
s1(s1 >= 2048) = s1(s1 >= 2048) - 4096;
sel = find(strcmpi(names, 'MLII'), 1); if isempty(sel), sel = 1; end
if sel == 1, s = s0; else, s = s1; end
rec.lead_name = names{sel};
rec.ecg_mv = ((s(:) - zero(sel)) / gain(sel));
rec.ann_s = zeros(0, 1); rec.ann_code = zeros(0, 1); rec.ann_cls = zeros(0, 1);
if exist([base '.atr'], 'file')
    [samp, code] = read_atr([base '.atr']);
    cls = aami_label(code);
    keep = cls > 0;
    rec.ann_s = samp(keep); rec.ann_code = code(keep); rec.ann_cls = cls(keep);
end
end

function [samp, code] = read_atr(file)
fid = fopen(file, 'r'); b = fread(fid, inf, 'uint8=>double'); fclose(fid);
n = numel(b); i = 1; t = 0;
samp = zeros(0, 1); code = zeros(0, 1);
while i + 1 <= n
    w = b(i) + 256 * b(i + 1); i = i + 2;
    if w == 0, break; end
    A = floor(w / 1024); I = mod(w, 1024);
    if A == 59                       % SKIP: 32-bit interval follows
        hi = b(i) + 256 * b(i + 1); lo = b(i + 2) + 256 * b(i + 3); i = i + 4;
        sk = hi * 65536 + lo; if sk >= 2^31, sk = sk - 2^32; end
        t = t + sk;
    elseif A == 60 || A == 61 || A == 62   % NUM / SUB / CHN: no payload
        continue;
    elseif A == 63                   % AUX: I bytes (padded to even)
        i = i + I + mod(I, 2);
    else
        t = t + I;
        samp(end + 1, 1) = t; code(end + 1, 1) = A; %#ok<AGROW>
    end
end
end
