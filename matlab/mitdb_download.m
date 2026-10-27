function mitdb_download(data_dir, ids)
% MITDB_DOWNLOAD  Fetch .hea/.dat/.atr of MIT-BIH records from PhysioNet.
%   Needs internet access. Files already present are skipped.
if nargin < 1, data_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data'); end
if nargin < 2, ids = mitdb_split('all'); end
if ~exist(data_dir, 'dir'), mkdir(data_dir); end
base = 'https://physionet.org/files/mitdb/1.0.0/';
ext  = {'.hea', '.dat', '.atr'};
for i = 1:numel(ids)
    for k = 1:3
        f = fullfile(data_dir, sprintf('%d%s', ids(i), ext{k}));
        if exist(f, 'file') && dir(f).bytes > 0, continue; end
        url = sprintf('%s%d%s', base, ids(i), ext{k});
        fprintf('  downloading %s\n', url);
        if exist('websave', 'file') == 2 || exist('websave', 'builtin') == 5
            websave(f, url);
        else
            urlwrite(url, f); %#ok<URLWR>
        end
    end
end
end
