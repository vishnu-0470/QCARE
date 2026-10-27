function ref = ref_update(ref, x, nupd, cfg)
% REF_UPDATE  Stochastic-median tracker (no divider, no sort):
%   ref += sign(x-ref)*max(1, ref>>SH), SH = warm-up shift for the first
%   ref_warm_n updates, steady-state shift afterwards. ref==0 means "unset".
e = cfg.eng;
x = max(x, 0);
if ref == 0, ref = max(x, 1); return; end
if nupd < e.ref_warm_n, sh = e.ref_sh_warm; else, sh = e.ref_sh_run; end
step = max(1, floor(ref / 2^sh));
if x > ref, ref = ref + step;
elseif x < ref, ref = max(1, ref - step); end
end
