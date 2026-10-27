function FQ = fv_to_fq(FV)
% FV_TO_FQ  Features -> the 8 int8 side inputs of the CNN (same as beat_decide/RTL).
FQ = zeros(size(FV, 1), 8);
for j = 1:8
    if j == 7, FQ(:, j) = FV(:, 7) * 64;
    else, FQ(:, j) = min(max(floor((FV(:, j) - 256) / 4), -128), 127); end
end
end
