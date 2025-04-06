function [D, x] = fourier_diff_matrix_vpa(N, precision_digits)
% High-precision Fourier differentiation matrix using Symbolic Toolbox

digitsOld = digits();  % store old setting
digits(precision_digits);        % set desired precision

if mod(N, 2) == 0
    % EVEN N: cotangent-based
    x = sym(2 * pi) / N * (0:N-1)';
    D = sym(zeros(N));
    for i = 1:N
        for j = 1:N
            if i ~= j
                D(i, j) = 0.5 * (-1)^(i + j) * cot((x(i) - x(j)) / 2);
            end
        end
    end
else
   % ODD N: sine-based method (with correct sine grid)
    Nsym = sym(N);  % ensure symbolic
    x = pi * (1:N)' / (Nsym + 1);  % correct sine grid in [0, pi]
    D = sym(zeros(N));
    for i = 1:N
        for j = 1:N
            if i ~= j
                D(i, j) = (-1)^(i + j) / (2 * sin((i - j) * pi / (Nsym + 1)));
            end
        end
    end
    
    % Scale for periodic domain [0, 2pi] if needed
    x = 2 * x;         % map from [0, pi] to [0, 2pi]
    D = D * (2 / (2 * pi / (N + 1)));  % rescale to match dx
end

digits(digitsOld);  % restore previous setting
end

k_vals = 2:2:12;
tol = vpa(1e-5);
precision_digits = 100;

fprintf('Target: max relative error < %.0e using %d-digit precision\n\n', double(tol), precision_digits);

for k = k_vals
    fprintf('Testing k = %d\n', k);
    found = false;

    for N = 9:4:129  % reduce upper limit to keep things fast
        [D, x] = fourier_diff_matrix_vpa(N, precision_digits);

        u = exp(k * sin(x));
        du_true = k * cos(x) .* exp(k * sin(x));
        du_num = D * u;

        rel_error = abs((du_num - du_true) ./ du_true);
        max_rel_err = max(rel_error);

        fprintf('  N = %3d | max relative error = %.2e\n', N, double(max_rel_err));

        if max_rel_err < tol
            fprintf('  ✅ Acceptable: N = %d meets error requirement for k = %d\n\n', N, k);
            found = true;
            break;
        end
    end

    if ~found
        fprintf('  ❌ No N found up to 128 for k = %d\n\n', k);
    end
end