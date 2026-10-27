function [seg, r_pos, truncated] = paper_segment(x, Rp, R, Rn, b, target_len)
% PAPER_SEGMENT  Beat segmentation rule of Cao et al., Eq. (1):
%     start = Rp + b,    stop = Rn - (120 - b)
%   where Rp, R, Rn are the (1-based) R-peak indices of the previous, current and
%   next beat, and b is the beat-range bias (40..80; default 60).  The segment is
%   truncated to target_len (360) or zero-padded at the end, as in Sec. II-A.
%   r_pos is the 1-based position of R inside seg (NaN if the record edge is hit).
if nargin < 5 || isempty(b), b = 60; end
if nargin < 6, target_len = 360; end
start = Rp + b;
stop  = Rn - (120 - b);
if start < 1 || stop > numel(x) || stop < start
    error('paper_segment: beat window falls outside the record');
end
idx = start:stop;
L = min(numel(idx), target_len);
seg = zeros(1, target_len);
seg(1:L) = x(idx(1:L));
truncated = numel(idx) > target_len;
r_pos = R - start + 1;
if r_pos < 1 || r_pos > target_len, r_pos = NaN; end
end
