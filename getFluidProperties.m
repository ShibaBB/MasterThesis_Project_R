function [thickness, temperature_K, pressure, rel_humidity, density_humid_air, Cp, Cv, eta, gamma, c, kappa, Pr] = getFluidProperties(fiberfolder, porosityfolder)
    % Check if fiberfolder or porosityfolder is empty
    if isempty(fiberfolder) || isempty(porosityfolder)
        error('Fiber folder or Porosity folder is empty. Please provide valid folder names.');
    end
    
    % Construct the XML file name based on the folder names
    xmlFileName = [fiberfolder, '_', porosityfolder, '.aed1001.xml'];  % For example, 'Acrylic_92.aed1001.xml'
    
    % Construct the full file path for the XML using fullfile to prevent issues with slashes
    xmlFilePath = fullfile(pwd, fiberfolder, porosityfolder, xmlFileName);
    
    % Debug: Display constructed file path for verification
    disp(['Constructed XML file path: ', xmlFilePath]);

    % Ensure the file exists
    if ~exist(xmlFilePath, 'file')
        error('❌ XML file not found: %s', xmlFilePath);
    end
    
    % If file exists, read the XML content
    disp(['Reading XML file: ', xmlFilePath]);  % For debugging
    
    % Read the XML file into a DOM (Document Object Model)
    try
        xmlDoc = xmlread(xmlFilePath);
    catch
        error('❌ Failed to read XML file. Check if the XML file is valid.');
    end
    
    % Extract the required data from the XML
    % Sample thickness
    thicknessNode = xmlDoc.getElementsByTagName('SampleThickness');
    thickness = str2double(char(thicknessNode.item(0).getFirstChild.getData)); % thickness unit in xml file is mm 

    % Temperature
    temperatureNode = xmlDoc.getElementsByTagName('Temperature');
    temperature_C = str2double(char(temperatureNode.item(0).getFirstChild.getData));
    temperature_K = temperature_C + 273.15;  % Convert to Kelvin
    disp(['Temperature in Kelvin: ', num2str(temperature_K)]);  % Debugging print
    
    % Air pressure
    pressureNode = xmlDoc.getElementsByTagName('AirPressure');
    pressure = str2double(char(pressureNode.item(0).getFirstChild.getData)) * 1000;  % Convert to Pa

    % Relative humidity
    humidityNode = xmlDoc.getElementsByTagName('RelativeAirHumidity');
    rel_humidity = str2double(char(humidityNode.item(0).getFirstChild.getData)) * 0.01;  % Convert to decimal

    % Calculate the saturated vapor pressure and other thermodynamic properties based on temperature
    if temperature_K >= 304 && temperature_K < 334
        A = 5.20389;
        B = 1733.926;
        C = -39.485;
    elseif temperature_K >= 273 && temperature_K < 304
        A = 5.40221;
        B = 1838.675;
        C = -31.737;
    else
        error('Temperature not in valid region for Antoine equation coefficients');
    end

    % Antoine Equation to calculate the saturated vapor pressure in bar
    saturated_vapor_pressure = 10^(A - (B / (temperature_K + C)));  % in bar
    sat_vap_pres_mbar = saturated_vapor_pressure * 1e3;  % Convert to mbar

    % Calculate the mass fraction of water vapor and psi (dimensionless)
    x_s = 0.622 * (sat_vap_pres_mbar / (pressure - sat_vap_pres_mbar));  % in kg/kg
    psi = (rel_humidity * (pressure - sat_vap_pres_mbar)) / (pressure - rel_humidity * sat_vap_pres_mbar);

    % Calculate the mixing ratio of water vapor in humid air
    x = psi * x_s;  % in kg/kg

    % Constants for the gas equations
    R_air = 287.06;  % Specific gas constant for dry air [J/kg K]
    R_steam = 461.52;  % Specific gas constant for steam [J/kg K]

    % Calculate the density of humid air using the equation for ideal gas
    density_humid_air = (pressure / (R_steam * temperature_K)) * ((1 + x) / (0.622 + x));  % Convert pressure to Pa, density calculation

    % Calculate specific heat at constant pressure (Cp), volume (Cv), dynamic viscosity (eta), and gamma
    Cp = 4168.8 * (0.249679 - 7.55179e-5 * temperature_K + 1.69194e-7 * temperature_K^2 - 6.46128e-11 * temperature_K^3);
    Cv = Cp - R_air;  % Specific heat at constant volume
    eta = 7.72488e-8 * temperature_K - 5.95238e-11 * temperature_K^2 + 2.71368e-14 * temperature_K^3;  % Dynamic viscosity
    gamma = Cp / Cv;  % Ratio of specific heats (gamma)

    % Calculate velocity of sound in the humid air
    c = sqrt(gamma * R_air * temperature_K);  % Speed of sound in m/s

    % Debugging print for kappa calculation
    disp(['Temperature (K) for kappa calculation: ', num2str(temperature_K)]);

    % Calculate thermal conductivity (kappa) for humid air
    try
        kappa = 2.624e-02 * ((temperature_K / 300)^(3 / 2) * (300 + 245.4 * exp(-27.6 / 300)) / (temperature_K + 245.4 * exp(-27.6 / temperature_K)));
        disp(['Calculated kappa: ', num2str(kappa)]);
    catch
        warning('Failed to calculate kappa. Check the temperature value and formula.');
        kappa = NaN;  % Assign NaN if there's an error in the calculation
    end

    % Calculate the Prandtl number
    Pr = eta * Cp / kappa;

    % Return all calculated values
end
