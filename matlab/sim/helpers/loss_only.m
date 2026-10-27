function l = loss_only(X, y, P)
lg = irn_net_forward(X, P);
[l] = nn_softmax_ce(lg, y);
end
