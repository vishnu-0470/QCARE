function p = afe_default()
% AFE_DEFAULT  AFE-twin parameters used for the experiments.  EVERY value is a
%   placeholder until calibrated against a circuit simulation in Cadence Spectre
%   (gain from AC at 10 Hz, corners from AC, noise from Spectre 'noise',
%   rails from an overdriven transient, recovery from an offset-step run).
p = struct('gain_dB', 38, 'f_hp', 0.5, 'f_lp', 150, 'v_clip_pos', 0.6887, 'v_clip_neg', -0.7945, ...
           'bits', 12, 'noise_uVrms', 5, 'mains_uV', 0, 'offset_mV', 0, 'rec_slow', 3, 'seed', 1);
end
