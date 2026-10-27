function [g, tau] = cwt_kernel(j, omega)
% CWT_KERNEL  Cosine-approximated Morlet wavelet at scale j (Cao et al., Eq. 6-8).
%   g(tau) = (1/sqrt(j)) * cos(omega * tau / j) * exp(-(tau / j)^2 / 2),
%   evaluated on the integer lags tau = -4j .. 4j (the Gaussian is e^-8 at the
%   ends, so the truncation is negligible).  Both returned as row vectors.
L   = 4 * j;
tau = -L:L;
g   = (1 / sqrt(j)) * cos(omega * tau / j) .* exp(-(tau / j) .^ 2 / 2);
end
