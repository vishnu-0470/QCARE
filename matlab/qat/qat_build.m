function Qm = qat_build(P, E, eWfix)
% If eWfix (struct with fields named like the layers) is given, the weight
% exponents are frozen to those values instead of being recomputed from P.
% QAT_BUILD  Quantise the latent float network P to the integer model used by
%   the hardware.  E holds the activation exponents from qat_calibrate.
%   Every scale is a power of two; every bias is stored in its accumulator scale.
Qm.e_in = -7;                  % input map: int8 with real value q * 2^-7
Qm.E = E;
if nargin < 3, eWfix = []; end

[Wq, eW] = wq(P.ds1.W, eWfix, 'ds1');
eA = Qm.e_in + eW;
Qm.ds1 = struct('W', Wq, 'eW', eW, 'b', bias_q(P.ds1.b, eA), 'eA', eA);

Qm.irn1 = block_q(P.irn1, E.ds1, E.blk1, eWfix, 'irn1');
Qm.irn2 = block_q(P.irn2, E.blk1, E.blk2, eWfix, 'irn2');

[Wq, eW] = wq(P.ds2.W, eWfix, 'ds2');
eA = E.blk2 + eW;
Qm.ds2 = struct('W', Wq, 'eW', eW, 'b', bias_q(P.ds2.b, eA), 'eA', eA);

Qm.irn3 = block_q(P.irn3, E.ds2, E.blk3, eWfix, 'irn3');
Qm.irn4 = block_q(P.irn4, E.blk3, E.blk4, eWfix, 'irn4');

[Wq, eW] = wq(P.fc.W, eWfix, 'fc');
eA = E.blk4 + eW;
Qm.fc = struct('W', Wq, 'eW', eW, 'b', bias_q(P.fc.b, eA), 'eA', eA);
end

function B = block_q(Pb, ex, eb, eWfix, nm)
% one Inception-ResNeXt block: input exponent ex, output/internal exponent eb
[B.W1, B.eW1] = wq(Pb.W1, eWfix, [nm '_W1']);
[B.W2, B.eW2] = wq(Pb.W2, eWfix, [nm '_W2']);
[B.W3, B.eW3] = wq(Pb.W3, eWfix, [nm '_W3']);
B.ex = ex; B.eb = eb;
B.eA1 = ex + B.eW1;  B.b1 = bias_q(Pb.b1, B.eA1);
B.eA2 = eb + B.eW2;  B.b2 = bias_q(Pb.b2, B.eA2);
B.eA3 = eb + B.eW3;  B.b3 = bias_q(Pb.b3, B.eA3);
end

function bq = bias_q(b, eA)
bq = floor(b(:) / 2^eA + 0.5);
assert(all(abs(bq) <= 32767), 'qat_build: bias does not fit 16 bits');
end

function [q, e] = wq(W, eWfix, key)
% weight quantisation with optional frozen exponent (key = 'layer' or 'layer.Wk')
if isempty(eWfix)
    [q, e] = qat_weight_quant(W);
else
    e = eWfix.(key);
    q = min(max(floor(W / 2^e + 0.5), -8), 7);
end
end
