function [loss, dz, p] = nn_softmax_ce(z, y)
% NN_SOFTMAX_CE  Softmax cross-entropy for one sample.  z [K 1], y in 1..K.
m = max(z);
e = exp(z - m);
p = e / sum(e);
loss = -log(p(y) + 1e-300);
dz = p;
dz(y) = dz(y) - 1;
end
