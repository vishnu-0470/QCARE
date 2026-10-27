function [X, U, FQ, Y, PID] = gather_set(Plist, Dlist)
% GATHER_SET  Stack the usable (not dropped, matched N/S/V) beats of several records.
X = zeros(0, 10); U = zeros(0, 120); FQ = zeros(0, 8); Y = zeros(0, 1); PID = zeros(0, 1);
for r = 1:numel(Plist)
    P = Plist{r}; D = Dlist{r}; if isempty(P.S.raw), continue; end
    ok = ~D.drop & P.lbl >= 1 & P.lbl <= 3;
    if ~any(ok), continue; end
    X = [X; D.fv(ok, :)]; U = [U; reshape([P.S.raw(ok).u], cfg_L(), []).']; %#ok<AGROW>
    FQ = [FQ; fv_to_fq(D.fv(ok, :))]; Y = [Y; P.lbl(ok) - 1]; PID = [PID; P.id * ones(sum(ok), 1)]; %#ok<AGROW>
end
end
function L = cfg_L(), L = 120; end
