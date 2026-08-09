% Constraint function to ensure each assembly does not exceed 80 mm
function [c, ceq] = thickness_constraint_parallel(params, configuration)
    % Each assembly must not exceed 80 mm individually
    switch configuration
        case 1  % Fiber + Air
            max_thickness = max(params(1), params(3));  % Maximum of fiber or air thickness
        case 2  % Fiber + Fiber
            max_thickness = max(params(1), params(3));  % Maximum of two fiber layers
        case 3  % MPP + Fiber
            max_thickness = max(params(1), params(5));  % Maximum of fiber or MPP
        otherwise
            error('Invalid configuration');
    end
    
    % Inequality constraints: Thickness should be <= 80 mm (0.08m)
    c = max_thickness - 80e-3;  
    ceq = []; % No equality constraints
end
