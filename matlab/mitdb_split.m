function ids = mitdb_split(which_set)
% MITDB_SPLIT  De Chazal et al. (2004) inter-patient split of MIT-BIH.
%   Paced records 102,104,107,217 are excluded (AAMI EC57).
%   Cross-check these lists against the original paper before publishing.
DS1 = [101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230];
DS2 = [100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234];
switch lower(which_set)
    case 'ds1', ids = DS1;
    case 'ds2', ids = DS2;
    case 'all', ids = sort([DS1 DS2]);
    otherwise, error('mitdb_split: use ds1 | ds2 | all');
end
end
