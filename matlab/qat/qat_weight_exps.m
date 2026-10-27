function eW = qat_weight_exps(P)
% QAT_WEIGHT_EXPS  Weight exponents of every quantised tensor, from latent P.
%   Field names use underscores (MATLAB does not allow dots in field names).
%   Keys: ds1, ds2, fc, and irnK_W1 / irnK_W2 / irnK_W3 for K = 1..4.
[~, eW.ds1] = qat_weight_quant(P.ds1.W);
[~, eW.ds2] = qat_weight_quant(P.ds2.W);
[~, eW.fc]  = qat_weight_quant(P.fc.W);
for k = 1:4
    nm = sprintf('irn%d', k);
    [~, eW.([nm '_W1'])] = qat_weight_quant(P.(nm).W1);
    [~, eW.([nm '_W2'])] = qat_weight_quant(P.(nm).W2);
    [~, eW.([nm '_W3'])] = qat_weight_quant(P.(nm).W3);
end
end