function u_all = rk4_solver_matrix(N, dt, steps, method, precision_digits)
    % Solves du/dt = -2π du/dx using RK4 with finite difference or Fourier matrix

    % === Grid and initial condition ===
    L = 2*pi;
    x = (0:N-1)' * (L / N);  % Exactly N points, periodic
    u = exp(sin(x));         % Initial condition
    u_all = zeros(N, steps+1);
    u_all(:,1) = u;

    % === Setup differentiation operator ===
    if strcmp(method, 'fourier')
        [D, ~] = fourier_diff_matrix_vpa(N, precision_digits, 'odd');
        D = double(D);
    else
        dx = x(2) - x(1);
    end

    % === Time stepping ===
    for n = 1:steps
        F = @(u_in) compute_rhs(u_in);

        u1 = u + 0.5 * dt * F(u);
        u2 = u + 0.5 * dt * F(u1);
        u3 = u + dt * F(u2);

        u = (1/3) * (-u + u1 + 2*u2 + u3 + 0.5 * dt * F(u3));
        u_all(:,n+1) = u;
    end

    % === RHS function ===
    function rhs = compute_rhs(u)
        switch method
            case 'fd2'
                rhs = -2*pi * (circshift(u, -1) - circshift(u, 1)) / (2*dx);
            case 'fd4'
                rhs = -2*pi * (-circshift(u, 2) + 8*circshift(u, 1) ...
                             - 8*circshift(u, -1) + circshift(u, -2)) / (12*dx);
            case 'fourier'
                rhs = -2*pi * (D * u);
            otherwise
                error('Unknown method');
        end
    end
end
