function names = flat_names(P)
names = {};
for a = {'ds1', 'irn1', 'irn2', 'ds2', 'irn3', 'irn4', 'fc'}
    sub = P.(a{1});
    for f = fieldnames(sub).'
        names{end + 1} = [a{1} '.' f{1}];
    end
end
