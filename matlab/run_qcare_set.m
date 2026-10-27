function run_qcare_set(set_name)
%RUN_QCARE_SET Run a named MATLAB workflow from the Q-CARE project.
%   run_qcare_set('verify')   Compare the frozen model and export vectors.
%   run_qcare_set('waveform') Convert data/2.csv into a CWT/QNN input map.
%   run_qcare_set('stages')   Run the CWT, network, QAT, and export checks.

if nargin < 1 || isempty(set_name)
    fprintf('Choose a MATLAB set:\n');
    fprintf('  run_qcare_set(''verify'')   - compare ten frozen vectors\n');
    fprintf('  run_qcare_set(''waveform'') - process data/2.csv into a CWT map\n');
    fprintf('  run_qcare_set(''stages'')   - run stage 1 through stage 4 checks\n');
    return;
end

matlab_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(matlab_dir);
original_dir = pwd;
restore_dir = onCleanup(@() cd(original_dir)); %#ok<NASGU>
addpath(genpath(matlab_dir));
cd(project_root);

set_name = lower(char(set_name));
switch set_name
    case {'verify','verification'}
        verify_current_export();

    case {'waveform','cwt'}
        afe_output_root = fullfile(project_root,'experiments','afe_adc_matlab'); %#ok<NASGU>
        run(fullfile(matlab_dir,'sim','prepare_afe_qnn_input.m'));

    case {'stages','stage-checks'}
        scripts = {'t_stage1.m','t_stage2.m','t_stage3.m','t_stage4_export.m'};
        for k = 1:numel(scripts)
            fprintf('\n=== MATLAB %s ===\n', scripts{k});
            run(fullfile(matlab_dir,'sim',scripts{k}));
        end

    otherwise
        error('Unknown MATLAB set "%s". Use verify, waveform, or stages.', set_name);
end
end
