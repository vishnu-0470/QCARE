function Mi = quantize_map_int8(M)
% Canonical QNN input scale e_in=-7, shared with QAT, export and Python.
% Negative half ties round toward +infinity. The archived helper used 127.
Mi = int8(max(min(floor(128 * double(M) + 0.5), 127), -128));
end
