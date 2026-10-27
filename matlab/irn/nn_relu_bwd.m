function dX = nn_relu_bwd(dY, Z)
% NN_RELU_BWD  Gradient of ReLU given its pre-activation Z.
dX = dY .* (Z > 0);
end
