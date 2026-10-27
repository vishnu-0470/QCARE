function [dv, dW, db] = nn_fc_bwd(dz, v, W)
% NN_FC_BWD  Gradients of nn_fc_fwd.
dv = W.' * dz;
dW = dz * v.';
db = dz;
end
