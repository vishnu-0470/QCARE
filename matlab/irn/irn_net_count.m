function n = irn_net_count(P)
% IRN_NET_COUNT  Total number of trainable parameters.
n = 0;
fn = fieldnames(P);
for i = 1:numel(fn)
    sub = P.(fn{i});
    sf = fieldnames(sub);
    for k = 1:numel(sf), n = n + numel(sub.(sf{k})); end
end
end
