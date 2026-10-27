function cache = qat_real_cache(C, Qm)
% QAT_REAL_CACHE  Convert integer forward tensors to the real-valued cache that
%   irn_net_backward expects (real = q * 2^exponent).  Signs are preserved, so
%   ReLU masks are identical to the integer model.
E = Qm.E;
cache.X  = C.qin * 2^Qm.e_in;
cache.z1 = C.acc_ds1 * 2^(Qm.e_in + Qm.ds1.eW);
cache.c1 = block_cache(C.cb{1}, Qm.irn1);
cache.c2 = block_cache(C.cb{2}, Qm.irn2);
cache.pc = C.pc; cache.psz = C.psz;
cache.z2 = C.acc_ds2 * 2^(E.blk2 + Qm.ds2.eW);
cache.p  = C.P2 * 2^E.blk2;
cache.c3 = block_cache(C.cb{3}, Qm.irn3);
cache.c4 = block_cache(C.cb{4}, Qm.irn4);
cache.v4sz = C.v4sz;
cache.g = C.g * 2^E.blk4;
end

function c = block_cache(cb, B)
c.x  = cb.x * 2^B.ex;
c.a1 = cb.acc1 * 2^B.eA1;
c.A  = cb.A * 2^B.eb;
c.a2 = cb.acc2 * 2^B.eA2;
c.B1 = cb.B1 * 2^B.eb;
c.a3 = cb.acc3 * 2^B.eA3;
c.Z  = cb.Z * 2^B.eb;
end
