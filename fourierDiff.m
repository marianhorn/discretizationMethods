function fourier_diff_driver()
    k_vals = 2:2:12;
    tol = vpa(1e-5);
    precision_digits = 100;

    method = 'odd';  % 'even' or 'odd'

    fprintf('Target: max relative error < %.0e using %d-digit precision (%s method)\n\n', ...
        double(tol), precision_digits, method);

    for k = k_vals
        fprintf('Testing k = %d\n', k);
        found = false;

        for N = 8:4:128
            [D, x] = fourier_diff_matrix_vpa(N, precision_digits, method);

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
end


function [D, x] = fourier_diff_matrix_vpa(N, precision_digits, method)
% High-precision Fourier differentiation matrix using Symbolic Toolbox
% method: 'even' for cotangent method with N points
%         'odd'  for sine-based method with N+1 points

digitsOld = digits();
digits(precision_digits);

if strcmp(method, 'even')
    % EVEN N: cotangent-based (standard periodic)
    x = sym(2 * pi) / N * (0:N-1)';
    D = sym(zeros(N));
    for i = 1:N
        for j = 1:N
            if i ~= j
                D(i, j) = 0.5 * (-1)^(i + j) * cot((x(i) - x(j)) / 2);
            end
        end
    end

elseif strcmp(method, 'odd')
    % ODD case: sine-based with N+1 points
    Nsym = sym(N + 1);  % adjust grid size
    h = 2 * pi / Nsym;
    x = h * (0:N)';  % grid from 0 to 2pi (N+1 points)

    D = sym(zeros(N + 1));
    for j = 0:N
        for i = 0:N
            if i ~= j
                sign_factor = (-1)^(j + i);
                angle = (j - i) * pi / Nsym;
                D(j + 1, i + 1) = sign_factor / (2 * sin(angle));
            end
        end
    end

else
    error('Unknown method: %s. Use ''even'' or ''odd''.', method);
end

digits(digitsOld);
end
