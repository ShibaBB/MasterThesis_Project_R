% Constraint function to ensure total thickness is 80 mm
function [c, ceq] = thickness_constraint(params, configuration)
    % Calculate total thickness based on configuration
    switch configuration
        case 1  % Fiber + Air
            total_thickness = params(1) + params(3);  % Fiber + Air
        case 2  % Fiber + Fiber
            total_thickness = params(1) + params(3);  % Fiber 1 + Fiber 2
        case 3  % MPP + Fiber
            total_thickness = params(1) + params(5);  % Fiber + MPP
        otherwise
            error('Invalid configuration');
    end
    
    % Inequality constraints: Deviations from the 80 mm target
    c = total_thickness - 80e-3;  % total_thickness should be <= 80mm (0.08m)
    ceq = [];  % No equality constraints
end
