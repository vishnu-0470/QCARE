function E = qat_calibrate(P, qmaps)
% QAT_CALIBRATE  Choose power-of-two activation exponents from calibration data.
%   qmaps : [N x 1 x 60 x 60] integer input maps.  Use DS1 records only.
%   Runs the float network with real inputs (qmaps * 2^-7) and records the
%   99.9th-percentile activation at each quantisation point, per block.
vD1 = []; vB = {[], [], [], []}; vD2 = []; 
for n = 1:size(qmaps, 1)
    X = reshape(qmaps(n, 1, :, :), 1, 60, 60) * 2^-7;
    [~, c] = irn_net_forward(X, P);
    h1 = max(c.z1, 0);
    vD1 = [vD1; h1(:)];                                   %#ok<AGROW>
    [u1, cb1] = irn_block_fwd(h1, P.irn1);
    [u2, cb2] = irn_block_fwd(u1, P.irn2);
    vB{1} = [vB{1}; block_vals(cb1)];                     %#ok<AGROW>
    vB{2} = [vB{2}; block_vals(cb2)];                     %#ok<AGROW>
    [p2, ~] = nn_maxpool2_fwd(u2);
    h2 = max(c.z2, 0); vD2 = [vD2; h2(:)];                %#ok<AGROW>
    [u3, cb3] = irn_block_fwd(h2, P.irn3);
    [u4, cb4] = irn_block_fwd(u3, P.irn4);
    vB{3} = [vB{3}; block_vals(cb3)];                     %#ok<AGROW>
    vB{4} = [vB{4}; block_vals(cb4)];                     %#ok<AGROW>
end
E.ds1 = qat_exp_for_values(vD1);
E.blk1 = qat_exp_for_values(vB{1});
E.blk2 = qat_exp_for_values(vB{2});
E.ds2 = qat_exp_for_values(vD2);
E.blk3 = qat_exp_for_values(vB{3});
E.blk4 = qat_exp_for_values(vB{4});
end

function v = block_vals(cb)
v = [cb.A(:); max(cb.a2(:), 0); cb.B1(:); max(cb.a3(:), 0); max(cb.Z(:), 0)];
end
