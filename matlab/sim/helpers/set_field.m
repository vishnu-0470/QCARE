function S = set_field(S, path, q, dv)
parts = strsplit(path, '.');
v = S.(parts{1}).(parts{2});
v(q) = v(q) + dv;
S.(parts{1}).(parts{2}) = v;
end
