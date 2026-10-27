function dX = nn_gap_bwd(dv, sz)
% NN_GAP_BWD  Gradient of global average pooling; sz = [C H W].
HW = sz(2) * sz(3);
dX = reshape(repmat(dv / HW, 1, HW), sz);
end
