function v = get_field(S, path)
parts = strsplit(path, '.');
v = S.(parts{1}).(parts{2});
end
