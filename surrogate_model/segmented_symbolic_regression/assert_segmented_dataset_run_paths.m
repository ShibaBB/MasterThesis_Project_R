function assert_segmented_dataset_run_paths(dataset_run, varargin)
%ASSERT_SEGMENTED_DATASET_RUN_PATHS Reject standard dataset paths from another run.

dataset_run = char(string(dataset_run));
for i = 1:numel(varargin)
    candidate_path = char(string(varargin{i}));
    tokens = regexp(candidate_path, '[\\/]datasets[\\/]([^\\/]+)[\\/]', 'tokens', 'once');
    if ~isempty(tokens) && ~strcmp(tokens{1}, dataset_run)
        error(['Dataset run mismatch: configured run is %s, but this path belongs ' ...
            'to %s: %s'], dataset_run, tokens{1}, candidate_path);
    end
end
end
