clear;

% === Parameters ===
N = 128;                   % grid points
L = 2*pi;                  % domain length
x = (0:N-1)' * (L / N);    % periodic grid: x_j = 2π j / N
dx = x(2) - x(1);
dt = 0.001;
T = 1.0;
steps = round(T / dt);

method = 'fourier';        % 'fd2', 'fd4', or 'fourier'
precision_digits = 50;     % only relevant for 'fourier'

% === Initial condition ===
u0 = exp(sin(x));

% === Run RK4 solver ===
u_all = rk4_solver_matrix(N, dt, steps, method, precision_digits);
u_final = u_all(:, end);

% === Exact solution at t = T ===
u_exact = exp(sin(x - 2*pi*T));

% === Error metrics ===
error_Linf = max(abs(u_final - u_exact));
error_L2 = sqrt(mean((u_final - u_exact).^2));

fprintf('--- RK4 Evaluation ---\n');
fprintf('Method: %s\n', method);
fprintf('Grid points N: %d\n', N);
fprintf('Final time T: %.2f\n', T);
fprintf('L-infinity error: %.3e\n', error_Linf);
fprintf('L2 error:         %.3e\n', error_L2);

% === Plotting ===
figure;
plot(x, u_final, 'b-', x, u_exact, 'r--');
legend('Numerical', 'Exact');
title(sprintf('RK4 Solution vs Exact at t = %.2f', T));
xlabel('x'); ylabel('u(x)');
grid on;
