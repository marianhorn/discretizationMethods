% Exact solution using standard double precision
function u = exact_solution(x, t)
    u = exp(sin(x - 2 * pi * t));
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

% Fourier differentiation matrix using optimized MATLAB matrix operations
function D = fourier_diff_matrix(N, dx)
    D = zeros(N, N);
    for j = 1:N
        for i = 1:N
            if i ~= j
                angle = (j - i) * pi / (N + 1);  % Adjusted angle term
                D(j, i) = (-1)^(j + i) / (2 * sin(angle));  % Correct formula
            end
        end
    end
end
% Time stepping using 4th order Runge-Kutta
function u = rk4_step(u, dt, derivative_func)
    u1 = u + (dt * derivative_func(u))/2;
    u2 = u + (dt * derivative_func(u1))/2;
    u3 = u + dt * derivative_func(u2);
    u4 = (1/3)*(-u+u1+2*u2+u3+(dt*derivative_func(u3)/2));
    u = u4;
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
        D = fourier_diff_matrix(N, dx);  % Correct Fourier matrix
        derivative = @(u) -2 * pi * (D * u');  % Matrix-vector multiplication
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
    % Ensure that the error is scalar by taking the max absolute difference
    error = max(abs(u_num - u_exact));
end


% Convergence test
N_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
%methods = {'fd2', 'fd4', 'fourier'};
methods = {'fourier'};

T = pi;
dt = 0.0001;

for m = 1:length(methods)
    method = methods{m};
    disp(['Method: ', method]);
    %parpool;  % Start a parallel pool of workers (uses all available cores by default)
    
    % Initialize the 'errors' array before the parallel loop
    errors = zeros(1, length(N_values));  
    
    for i = 1:length(N_values)
        N = N_values(i);
        [x, u_num, u_exact] = solve_pde(N, T, dt, method);
        error = compute_error(u_num, u_exact);
        errors(i) = error;  % Store the result in the pre-allocated 'errors' array
        fprintf('N = %-5d Error = %.3e\n', N, error);  % This can be slow in parallel but works
    end
    
    %delete(gcp);  % Close the parallel pool after computation is done

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
