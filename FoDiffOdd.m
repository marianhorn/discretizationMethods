function analyze_fourier_diff_odd_case()
    k_list = [2, 4, 6, 8, 10, 12];
    tol = vpa(1e-5);
    digits(100);  % set global precision

    maxN = 300;
    figure;
    hold on;
    minimal_N_map = containers.Map();

    for k = k_list
        N_vals = 2:2:maxN;
        max_errors = zeros(size(N_vals));

        for idx = 1:length(N_vals)
            N = N_vals(idx);
            [D, x] = build_odd_fourier_diff_matrix(N);

            u = exp(k * sin(x));
            u_exact = k * cos(x) .* exp(k * sin(x));
            u_approx = D * u;

            rel_errors = abs((u_approx - u_exact) ./ u_exact);
            max_errors(idx) = double(max(rel_errors));

            if mod(N, 10) == 0
                fprintf('[DEBUG] k=%d, N=%d: max_rel_error = %.3e\n', k, N, max_errors(idx));
            end
        end

        % Plot convergence curve
        semilogy(N_vals, max_errors, 'DisplayName', sprintf('k = %d', k));

        % Find minimal N
        idx_min = find(max_errors < double(tol), 1);
        if ~isempty(idx_min)
            minimal_N_map(num2str(k)) = N_vals(idx_min);
        else
            minimal_N_map(num2str(k)) = NaN;
        end
    end

    xlabel('N'); ylabel('Max Relative Error');
    title('Fourier Differentiation Error (Odd N, High Precision)');
    legend('Location', 'southwest');
    grid on;

    % Summary printout
    fprintf('\nSummary of minimal N for error < 1e-5:\n');
    keys = minimal_N_map.keys;
    for i = 1:length(keys)
        k = keys{i};
        minN = minimal_N_map(k);
        if isnan(minN)
            fprintf('k=%s: ❌ No N ≤ %d met the error threshold.\n', k, maxN);
        else
            fprintf('k=%s: ✅ Minimal N = %d\n', k, minN);
        end
    end
end

function [D, x] = build_odd_fourier_diff_matrix(N)
    % Builds the sine-based Fourier differentiation matrix for odd case
    pi_sym = vpa(pi);
    h = 2 * pi_sym / (N + 1);
    x = sym(zeros(N + 1, 1));
    D = sym(zeros(N + 1));

    for j = 0:N
        x(j + 1) = h * j;
    end

    for j = 0:N
        for i = 0:N
            if i ~= j
                sign_factor = (-1)^(j + i);
                angle = (j - i) * pi_sym / (N + 1);
                D(j + 1, i + 1) = vpa(sign_factor) / (2 * sin(angle));
            else
                D(j + 1, i + 1) = vpa(0);
            end
        end
    end
end
