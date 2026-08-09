function [lb, ub] = getFiberConstraints(fiberfolder, phi, airProperties)
    % Function to get the lower and upper bounds for parameters based on fiber type and porosity
    
    % Fiber diameter range for each fiber type
    if strcmp(fiberfolder, 'Acrylic')
        d1 = 10e-6; 
        d2 = 40e-6; 
    elseif strcmp(fiberfolder, 'Silk')
        d1 = 4e-6; 
        d2 = 15e-6; 
    elseif strcmp(fiberfolder, 'Wool')
        d1 = 16e-6; 
        d2 = 40e-6; 
    else
        error('Unknown fiber type: %s', fiberfolder);
    end
    

    % Dynamic viscosity (eta) from airProperties
    eta = airProperties.eta;  % Dynamic viscosity from airProperties (in Pa.s)

    lb = [16 * eta * (1 - phi) ./ (d2^2 * (-0.640 * log(1 - phi) + 0.263 - phi)), ...
          1, ...
          (8 * eta ./ ((phi) .* (16 * eta * (1 - phi) ./ (d1^2 * (-0.640 * log(1 - phi) + 0.263 - phi))))) .^ 0.5, ...
          2*(8 * eta ./ ((phi) .* (16 * eta * (1 - phi) ./ (d1^2 * (-0.640 * log(1 - phi) + 0.263 - phi))))) .^ 0.5, ...
          1e-10,...
          1e-6];
    
    ub = [16 * eta * (1 - phi) ./ (d1^2 * (-0.640 * log(1 - phi) + 0.263 - phi)), ...
          3 - phi, ...
          (8 * eta * (3 - phi) ./ ((phi) .* (16 * eta * (1 - phi) ./ (d2^2 * (-0.640 * log(1 - phi) + 0.263 - phi))))) .^ 0.5, ...
          2*(8 * eta * (3 - phi) ./ ((phi) .* (16 * eta * (1 - phi) ./ (d2^2 * (-0.640 * log(1 - phi) + 0.263 - phi))))) .^ 0.5, ...
          1e-8,...
          1];

end


