function c = cnn_export(P)
% CNN_EXPORT  Latent parameters -> integer model used by the golden model/RTL.
q = @(w) min(max(round(w), -8), 7);
c.w1 = q(P.W1f); c.w2 = q(P.W2f); c.wf1 = q(P.Wf1f); c.wf2 = q(P.Wf2f);
c.b1 = round(P.b1(:)); c.b2 = round(P.b2(:)); c.bf1 = round(P.bf1(:)); c.bf2 = round(P.bf2(:));
c.s1 = P.s1; c.s2 = P.s2; c.s3 = P.s3;
lim = @(v, nm) assert(all(abs(v(:)) <= 32767), ['cnn_export: ' nm ' exceeds int16']);
lim(c.b1, 'b1'); lim(c.b2, 'b2'); lim(c.bf1, 'bf1'); lim(c.bf2, 'bf2');
end
