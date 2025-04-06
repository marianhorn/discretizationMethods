function compare_fourier_diff_methods()
    k_vals = 2:2:12;
    precision_digits = 100;
    tol = 1e-5;
    maxN = 128;

    methods = {'even', 'odd'};
    colors = {'b', 'r'};  % for plotting
    styles = {'-', '--'};

    figure; hold on;
    results = struct();

    for m = 1:length(methods)
        method = methods{m};
        method_label = upper(method);
        fprintf('\n--- Running method: %s ---\n', method_label);

        res = run_fourier_diff_analysis(k_vals, precision_digits, tol, method, maxN);

        % Plot all curves
        for k = k_vals
            kstr = num2str(k);
            semilogy(res.N_vals(kstr), res.max_errors(kstr), ...
                styles{m}, 'Color', colors{m}, ...
                'DisplayName', sprintf('%s k = %d', method_label, k));
        end

        results.(method) = res;
    end

    xlabel('N'); ylabel('Max Relative Error');
    title('Fourier Spectral Differentiation: Even vs Odd Methods');
    legend('Location', 'southwest');
    grid on;

    % Print comparison summary
    fprintf('\n=== Summary of Minimal N for Error < %.0e ===\n', tol);
    fprintf('%6s | %10s | %10s\n', 'k', 'Even N', 'Odd N');
    fprintf('-----------------------------\n');
    for k = k_vals
        ke = results.even.minimal_N(num2str(k));
        ko = results.odd.minimal_N(num2str(k));
        fprintf('%6d | %10s | %10s\n', k, printN(ke), printN(ko));
    end
end

function res = run_fourier_diff_analysis(k_vals, precision_digits, tol, method, maxN)
    digitsOld = digits();
    digits(precision_digits);

    tol = vpa(tol);
    minimal_N = containers.Map();
    all_errors = containers.Map();
    all_N_vals = containers.Map();

    for k = k_vals
        found = false;

        if strcmp(method, 'even')
            N_range = 8:4:maxN;
        elseif strcmp(method, 'odd')
            N_range = 8:4:maxN;
        else
            error('Unknown method: %s. Use ''even'' or ''odd''.', method);
        end

        max_errors = zeros(size(N_range));

        for idx = 1:length(N_range)
            N = N_range(idx);
            [D, x] = fourier_diff_matrix_vpa(N, precision_digits, method);

            u = exp(k * sin(x));
            du_true = k * cos(x) .* exp(k * sin(x));
            du_num = D * u;

            rel_error = abs((du_num - du_true) ./ du_true);
            max_rel_err = double(max(rel_error));
            max_errors(idx) = max_rel_err;

            if mod(N, 10) == 0
                fprintf('[DEBUG] method=%s, k=%d, N=%d: max_rel_error = %.3e\n', ...
                        method, k, N, max_rel_err);
            end

            if ~found && max_rel_err < double(tol)
                minimal_N(num2str(k)) = N;
                found = true;
            end
        end

        if ~found
            minimal_N(num2str(k)) = NaN;
        end

        all_errors(num2str(k)) = max_errors;
        all_N_vals(num2str(k)) = N_range;
    end

    res.minimal_N = minimal_N;
    res.max_errors = all_errors;
    res.N_vals = all_N_vals;

    digits(digitsOld);
end


function [D, x] = fourier_diff_matrix_vpa(N, precision_digits, method)
    digitsOld = digits();
    digits(precision_digits);

    if strcmp(method, 'even')
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
        Nsym = sym(N + 1);
        h = 2 * pi / Nsym;
        x = h * (0:N)';
        D = sym(zeros(N + 1));
        for j = 0:N
            for i = 0:N
                if i ~= j
                    D(j + 1, i + 1) = (-1)^(i + j) / (2 * sin((j - i) * pi / Nsym));
                end
            end
        end
    else
        error('Unknown method: %s. Use ''even'' or ''odd''.', method);
    end

    digits(digitsOld);
end

function s = printN(N)
    if isnan(N)
        s = '—';
    else
        s = num2str(N);
    end
end
