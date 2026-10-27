function [l, G] = loss_and_grads(X, y, P)
[lg, c] = irn_net_forward(X, P);
[l, dz] = nn_softmax_ce(lg, y);
G = irn_net_backward(dz, c, P);
end
