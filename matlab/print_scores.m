function print_scores(M, title)
fprintf('\n=== %s ===\n', title);
fprintf('annotated beats %d | detection Se %.2f%%  +P %.2f%% | accuracy(N/S/V, F/Q as errors) %.2f%%\n', M.n_ann, 100*M.det_se, 100*M.det_ppv, 100*M.acc);
fprintf('class |   Se(%%)  +P(%%)   F1(%%)\n');
for c = 1:3, fprintf('  %s   | %7.2f %6.2f %7.2f\n', M.names{c}, 100*M.Se(c), 100*M.PPV(c), 100*M.F1(c)); end
fprintf('early-exit %.1f%% of classified beats | dropped by AFE gate %.2f%%\n', 100*M.early_frac, 100*M.drop_frac);
fprintf('confusion (rows true N,S,V,F,Q ; cols N,S,V,drop,missed):\n'); disp(M.conf);
end
