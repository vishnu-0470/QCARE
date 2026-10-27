function Pr = qat_real_params(Qm)
% QAT_REAL_PARAMS  Real-valued parameters implied by the integer model (for STE backward).
Pr.ds1.W = Qm.ds1.W * 2^Qm.ds1.eW;  Pr.ds1.b = Qm.ds1.b * 2^Qm.ds1.eA;
Pr.irn1 = block_real(Qm.irn1);
Pr.irn2 = block_real(Qm.irn2);
Pr.ds2.W = Qm.ds2.W * 2^Qm.ds2.eW;  Pr.ds2.b = Qm.ds2.b * 2^Qm.ds2.eA;
Pr.irn3 = block_real(Qm.irn3);
Pr.irn4 = block_real(Qm.irn4);
Pr.fc.W = Qm.fc.W * 2^Qm.fc.eW;     Pr.fc.b = Qm.fc.b * 2^Qm.fc.eA;
end

function B = block_real(Qb)
B.W1 = Qb.W1 * 2^Qb.eW1; B.b1 = Qb.b1 * 2^Qb.eA1;
B.W2 = Qb.W2 * 2^Qb.eW2; B.b2 = Qb.b2 * 2^Qb.eA2;
B.W3 = Qb.W3 * 2^Qb.eW3; B.b3 = Qb.b3 * 2^Qb.eA3;
end
