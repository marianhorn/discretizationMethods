

% Exact solution using high precision
function u = exact_solution(x, t)
    pi_val = vpa(pi);
    u = vpa(exp(sin(vpa(x) - 2 * pi_val * vpa(t))));
end

% Initial condition
function u = initial_condition(x)
    u = exact_solution(x, 0);
end

% Second order finite difference derivative
function d = derivative_fd2(u, dx)
    d = (circshift(u, -1) - circshift(u, 1)) / (2 * dx);
end

% Fourth order finite difference derivative
function d = derivative_fd4(u, dx)
    d = (-circshift(u, 2) + 8 * circshift(u, 1) - 8 * circshift(u, -1) + circshift(u, -2)) / (12 * dx);
end

% Fourier differentiation matrix using high precision
function D = fourier_diff_matrix_gmp(N)
    pi_val = vpa(pi);
    D = zeros(N, N, 'vpa');
    for j = 1:N
        for i = 1:N
            if i ~= j
                angle = (j - i) * pi_val / N;
                D(j, i) = (-1)^(j + i) / (2 * sin(angle));
            end
        end
    end
end

% Matrix-vector multiplication using high precision
function result = matvec_gmp(D, u)
    N = length(u);
    result = zeros(N, 1, 'vpa');
    for j = 1:N
        result(j) = sum(D(j, :) .* u(:));
    end
    result = -2 * vpa(pi) * result;
end

% Time stepping using 4th order Runge-Kutta
function u = rk4_step(u, dt, derivative_func)
    k1 = dt * derivative_func(u);
    k2 = dt * derivative_func(u + 0.5 * k1);
    k3 = dt * derivative_func(u + 0.5 * k2);
    k4 = dt * derivative_func(u + k3);
    u = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
end

% Solver
function [x, u, u_exact] = solve_pde(N, T, dt, method)
    x = linspace(0, 2 * pi, N);
    dx = x(2) - x(1);
    u = initial_condition(x);
    
    if strcmp(method, 'fd2')
        derivative = @(u) -2 * pi * derivative_fd2(u, dx);
    elseif strcmp(method, 'fd4')
        derivative = @(u) -2 * pi * derivative_fd4(u, dx);
    elseif strcmp(method, 'fourier')
        D = fourier_diff_matrix_gmp(N);
        derivative = @(u) matvec_gmp(D, u);
    else
        error('Unknown method');
    end

    t = 0;
    while t < T
        u = rk4_step(u, dt, derivative);
        t = t + dt;
    end
    u_exact = exact_solution(x, T);
end

% Compute L-infinity error
function error = compute_error(u_num, u_exact)
    error = max(abs(u_num - u_exact));
end

% Convergence test
N_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
methods = {'fd2', 'fd4', 'fourier'};
T = pi;
dt = 0.01;

for m = 1:length(methods)
    method = methods{m};
    disp(['Method: ', method]);
    parpool;  % Start a parallel pool of workers (uses all available cores by default)
    
    % Initialize the 'errors' array before the parallel loop
    errors = zeros(1, length(N_values));  
    
    parfor i = 1:length(N_values)
        N = N_values(i);
        [x, u_num, u_exact] = solve_pde(N, T, dt, method);
        error = compute_error(u_num, u_exact);
        errors(i) = error;  % Store the result in the pre-allocated 'errors' array
        fprintf('N = %-5d Error = %.3e\n', N, error);  % This can be slow in parallel but works
    end
    
    delete(gcp);  % Close the parallel pool after computation is done

    % Estimate convergence rates
    rates = [];
    for i = 2:length(errors)
        rate = log(errors(i) / errors(i-1)) / log(N_values(i) / N_values(i-1));
        rates = [rates, rate];
    end
    disp('Convergence rates:');
    for i = 1:length(rates)
        fprintf('N = %-5d Rate ≈ %.2f\n', N_values(i+1), rates(i));
    end
end
