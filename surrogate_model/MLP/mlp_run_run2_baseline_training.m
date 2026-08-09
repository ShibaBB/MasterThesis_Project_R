%% Compatibility Driver
% The legacy run2 dataset no longer belongs to this repository. Execute the
% active paired-target run1 driver so stale automation fails neither silently
% nor by selecting the superseded absorption target.

script_dir = fileparts(mfilename('fullpath'));
run(fullfile(script_dir, 'mlp_run_baseline_training.m'));
