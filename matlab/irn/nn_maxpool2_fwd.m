function [Y, A] = nn_maxpool2_fwd(X)
% NN_MAXPOOL2_FWD  2x2 max pooling, stride 2.  X [C H W] with H, W even.
%   A holds the index (1..4) of the winner in each window, for the backward pass.
[C, H, W] = size(X);
Ho = H / 2; Wo = W / 2;
S = zeros(C, Ho, Wo, 4);
S(:, :, :, 1) = X(:, 1:2:H, 1:2:W);
S(:, :, :, 2) = X(:, 2:2:H, 1:2:W);
S(:, :, :, 3) = X(:, 1:2:H, 2:2:W);
S(:, :, :, 4) = X(:, 2:2:H, 2:2:W);
[Ym, Am] = max(S, [], 4);
Y = reshape(Ym, C, Ho, Wo);
A = reshape(Am, C, Ho, Wo);
end
