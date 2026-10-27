function v = nn_gap_fwd(X)
% NN_GAP_FWD  Global average pooling [C H W] -> [C 1].
[C, H, W] = size(X);
v = reshape(mean(reshape(X, C, H * W), 2), C, 1);
end
