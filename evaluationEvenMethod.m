function analyze_fourier_convergence()
    % Initialize parallel pool if not already running
    if isempty(gcp('nocreate'))
        parpool();
    end

    precision_digits = 50;
    method = 'even';
    N_vals = 8:4:64;

    funcs = {@(x) cos(10*x), @(x) cos(x/2), @(x) x};
    dfuncs = {@(x) -10*sin(10*x), @(x) -0.5*sin(x/2), @(x) ones(size(x))};
    labels = {'cos(10x)', 'cos(x/2)', 'x'};

    figure;
    for f_idx = 1:length(funcs)
        f = funcs{f_idx};
        df = dfuncs{f_idx};
        label = labels{f_idx};

        fprintf('\n=== Processing function: %s ===\n', label);

        Linf_errors = sym(zeros(size(N_vals)));
        L2_errors = sym(zeros(size(N_vals)));

        for i = 1:length(N_vals)
            N = N_vals(i);
            fprintf('[DEBUG] f = %s, N = %d → building matrix...\n', label, N);
            [D, x] = fourier_diff_matrix_vpa(N, precision_digits, method);

            fx = f(x);
            dfx_true = df(x);
            dfx_approx = D * fx;

            err = abs(dfx_true - dfx_approx);
            Linf_errors(i) = max(err);
            L2_errors(i) = sqrt(sum(err.^2) / length(err));

            fprintf('[DEBUG]    Linf error = %.3e | L2 error = %.3e\n', ...
                double(Linf_errors(i)), double(L2_errors(i)));
        end

        subplot(1, 2, 1);
        semilogy(N_vals, double(L2_errors), '-o', 'DisplayName', labels{f_idx});
        hold on; xlabel('N'); ylabel('L2 error'); grid on; title('L2 Error');

        subplot(1, 2, 2);
        semilogy(N_vals, double(Linf_errors), '-s', 'DisplayName', labels{f_idx});
        hold on; xlabel('N'); ylabel('L∞ error'); grid on; title('L∞ Error');
    end

    subplot(1, 2, 1); legend('Location', 'southwest');
    subplot(1, 2, 2); legend('Location', 'southwest');

    % Print error table
    fprintf('\n%-12s | %-10s | %-15s | %-15s\n', 'Function', 'N', 'L∞ error', 'L2 error');
    fprintf('---------------------------------------------------------------\n');
    for f_idx = 1:length(funcs)
        label = labels{f_idx};
        for i = 1:length(N_vals)
            fprintf('%-12s | %4d        | %.3e         | %.3e\n', ...
                label, N_vals(i), double(Linf_errors(i)), double(L2_errors(i)));
        end
        fprintf('---------------------------------------------------------------\n');
    end

    % ✍️ Interpretation Section
    fprintf('\nInterpretation:\n');
    fprintf('- cos(10x): High frequency ⇒ needs more N to resolve accurately.\n');
    fprintf('- cos(x/2): Low frequency ⇒ converges fast with fewer points.\n');
    fprintf('- x: Not periodic ⇒ slow convergence due to discontinuity in periodic extension.\n');
    fprintf('  Try subtracting the linear part or using non-periodic methods to improve.\n');

    % Clean up parallel pool
    delete(gcp('nocreate'));
end
