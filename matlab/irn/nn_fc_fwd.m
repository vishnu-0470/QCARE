function z = nn_fc_fwd(v, W, b)
% NN_FC_FWD  Fully connected layer: z = W*v + b  (v [D 1], W [K D], b [K 1]).
z = W * v + b;
end
