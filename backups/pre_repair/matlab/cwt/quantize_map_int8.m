function Mi = quantize_map_int8(M)
% QUANTIZE_MAP_INT8  8-bit feature map (Cao et al. Fig. 1): M in [-1,1] -> int8.
Mi = int8(max(min(round(127 * M), 127), -128));
end
