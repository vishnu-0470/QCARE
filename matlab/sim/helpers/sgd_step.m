function P = sgd_step(P, G, lr)
for a = {'ds1', 'irn1', 'irn2', 'ds2', 'irn3', 'irn4', 'fc'}
    for f = fieldnames(P.(a{1})).'
        P.(a{1}).(f{1}) = P.(a{1}).(f{1}) - lr * G.(a{1}).(f{1});
    end
end
