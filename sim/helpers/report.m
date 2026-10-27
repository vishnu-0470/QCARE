function report(c, msg)
  if c, t = 'PASS'; else, t = 'FAIL'; end
  fprintf('%s  %s\n', t, msg);
end
