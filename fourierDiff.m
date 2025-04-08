function compare_fourier_diff_methods()
    k_vals = 2:2:12;
    precision_digits = 50;
    tol = 1e-5;
    maxN = 256;

    methods = {'even', 'odd'};
    colors = {'b', 'r'};
    styles = {'-', '--'};

    % Start parallel pool if not already running
    pool = gcp('nocreate');
    if isempty(pool)
        parpool();  % default: uses all available workers
    end
    figure; hold on;
    results = struct();
    
    % Generate a colormap for different k values
    k_colors = lines(length(k_vals));  % MATLAB's built-in distinct color set
    
    for m = 1:length(methods)
        method = methods{m};
        method_label = upper(method);
        style = styles{m};  % '-' for even, '--' for odd
        fprintf('\n--- Running method: %s ---\n', method_label);
    
        res = run_fourier_diff_analysis(k_vals, precision_digits, tol, method, maxN);
    
        for idx = 1:length(k_vals)
            k = k_vals(idx);
            kstr = num2str(k);
            color = k_colors(idx, :);
    
            semilogy(res.N_vals(kstr), res.max_errors(kstr), ...
                style, 'Color', color, ...
                'DisplayName', sprintf('%s, k = %d', method_label, k));
        end
    
        results.(method) = res;
    end
    
    xlabel('N');
    ylabel('Max Relative Error');
    title('Fourier Spectral Differentiation: Even vs Odd Methods');
    legend('Location', 'northeast');  % Top-right corner
    grid on;

    set(gca, 'YScale', 'log');

    ylim([1e-80, 1e10]);  % You can adjust this if needed

    % Summary
    fprintf('\n=== Summary of Minimal N for Error < %.0e ===\n', tol);
    fprintf('%6s | %10s | %10s\n', 'k', 'Even N', 'Odd N');
    fprintf('-----------------------------\n');
    for k = k_vals
        ke = results.even.minimal_N(num2str(k));
        ko = results.odd.minimal_N(num2str(k));
        fprintf('%6d | %10s | %10s\n', k, printN(ke), printN(ko));
    end

    % Shut down parallel pool after computation
    delete(gcp('nocreate'));
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
            N_range = 8:2:maxN;
        elseif strcmp(method, 'odd')
            N_range = 8:2:maxN;
        else
            error('Unknown method: %s. Use ''even'' or ''odd''.', method);
        end

        max_errors = zeros(size(N_range));
        min_N = NaN;

        % ✅ Parallelize over N values (coarse-grained parallelism)
        parfor idx = 1:length(N_range)
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
        end
        
        % After the parfor: find the first N that meets the tolerance
        min_N = NaN;
        for idx = 1:length(N_range)
            if max_errors(idx) < double(tol)
                min_N = N_range(idx);
                break;
            end
        end

        minimal_N(num2str(k)) = min_N;
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
                    D(i, j) = 0.5 * (-1)^(i + j) / tan((x(i) - x(j)) / 2);
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
