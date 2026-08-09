function [Zs_norm, Reflect, alpha, Re, Im, Zc, k] = JCAL_O(freq, h, phi, sigma, alpha_infin, lambda, lambda_prime, k0_prime)
rho_air=1.204; % air density, unit: kg/m^3
eta=1.846*10^-5; % ﻿dynamic viscosity of air
j=(-1)^0.5; % ﻿complex number
omega=2*pi*freq; % angular frequency
Pr=0.71; % ﻿Prandtl number for the ambient air fluid
gamma=1.4; % adiabatic constant
P_air=101.325*10^3; % unit: Pa
v_sound=344; % sound velocity in air
air_Zc=rho_air*v_sound;

% Dynamic density
rho_omega=(rho_air*alpha_infin./phi).*(1+(sigma*phi./(j*rho_air.*omega*alpha_infin))...
    .*(1+4*j*omega*alpha_infin.^2*eta*rho_air./(sigma^2*lambda^2*phi^2)).^0.5);

% Dynamic modulus
K_omega= (gamma * P_air/ phi) .*...
    (gamma - (gamma - 1) ./ (1 - (j * eta * phi) ./ (Pr * omega * k0_prime*rho_air) .*...
    (1 + (4 * j * omega * k0_prime^2 * Pr*rho_air) ./ (eta * lambda_prime^2  * phi^2)).^0.5)).^-1;

% Zwikker and Kosten
Zc=(rho_omega.*K_omega).^0.5;
k=omega.*(rho_omega./K_omega).^0.5;

% surface impedance
Zs=Zc.*coth(j*k*h);

% normalized surface impedance
Zs_norm=Zs/air_Zc;

% Reflection coefficient
Reflect=(Zs_norm-1)./(Zs_norm+1);

% surface impedance normalization
Re=real(Zs_norm); % normalized surface impedance real part
Im=imag(Zs_norm); % normalized surface impedance imaginary part

% predicted sound absorption
alpha=1-(abs(Reflect)).^2;
end