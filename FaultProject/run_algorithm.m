

%% ------------------------------
%% 1. Dataset and preprocessing
datasets = {'kc1.csv', 'pc1.csv'};
dataStruct = struct();

for i = 1:length(datasets)
    [X_train, y_train, X_test, y_test] = preprocess_dataset(datasets{i});

    % Convert label to numeric 
    if iscell(y_train)
        [y_train_num, ~, label_map] = grp2idx(y_train);
        [y_test_num, ~] = grp2idx(y_test);
    else
        y_train_num = y_train;
        y_test_num = y_test;
    end

    % Save preprocessing result
    dataStruct(i).name = datasets{i};
    dataStruct(i).X_train = X_train;
    dataStruct(i).y_train_num = y_train_num;
    dataStruct(i).X_test = X_test;
    dataStruct(i).y_test_num = y_test_num;
end

%% ------------------------------
%% 2. Choose algorithm
fprintf('Select algorithm to run:\n1 - BOA\n2 - GWO\n');
choice = input('Enter choice (1 or 2): ');

switch choice
    case 1
        algorithm = 'BOA';
    case 2
        algorithm = 'GWO';
    otherwise
        error('Invalid choice! Enter 1 or 2.');
end
fprintf('Algorithm selected: %s\n\n', algorithm);

%% ------------------------------
%% 3. Parameter algorithm
nPop = 20;       % Population size
Max_iter = 50;   % Max iterations

results = struct();

%% ------------------------------
%% 4. Loop each dataset
for d = 1:length(dataStruct)
    dataset_name = dataStruct(d).name;
    X_train = dataStruct(d).X_train;
    y_train_num = dataStruct(d).y_train_num;
    X_test  = dataStruct(d).X_test;
    y_test_num = dataStruct(d).y_test_num;

    fprintf('===============================\n');
    fprintf('Running %s on %s\n', algorithm, dataset_name);
    fprintf('===============================\n');

    %% === NEW: Enable 50 runs per dataset ===
    num_runs = 50;
    results(d).runs = struct();

    %% --------------------------
    %% 4a. Fitness function 
    dim = size(X_train,2);
    lb = 0;   
    ub = 1;   
    fobj = @(w) classification_error(w,X_train,y_train_num);

    %% === NEW: Start 50-run loop ===
    for run = 1:num_runs
        fprintf(' Run %d/%d...\n', run, num_runs);

        %% --------------------------
        %% 4b. Run BOA/GWO
        tic;
        switch algorithm
            case 'BOA'
                [fmin, best_pos, Convergence_curve] = BOA(nPop, Max_iter, lb, ub, dim, fobj);
            case 'GWO'
                if exist('GWO','file') == 2  % 2 = file exists
                    [fmin, best_pos, Convergence_curve] = GWO(nPop, Max_iter, lb, ub, dim, fobj);
                else
                    error(['GWO.m not found!']);
                end
        end
        time_taken = toc;

        %% --------------------------
        %% 4c. Prediction using best_pos
        selected_features = best_pos > 0.5;  % threshold 0.5
        X_train_sel = X_train(:, selected_features);
        X_test_sel  = X_test(:, selected_features);

        % kNN classifier 
        mdl = fitcknn(X_train_sel, y_train_num, 'NumNeighbors', 5);
        predictions = predict(mdl, X_test_sel);

        %% --------------------------
        %% 4d. metrics
        accuracy = sum(predictions == y_test_num)/length(y_test_num)*100;
        convergence_speed = find(Convergence_curve <= fmin,1,'first');
        search_capability = fmin;
        time_complexity = time_taken;

        %% === NEW: Save per-run results ===
        results(d).runs(run).accuracy = accuracy;
        results(d).runs(run).convergence_speed = convergence_speed;
        results(d).runs(run).search_capability = search_capability;
        results(d).runs(run).time_complexity = time_complexity;
        results(d).runs(run).Convergence_curve = Convergence_curve;
        results(d).runs(run).predictions = predictions;

    end
    %% === NEW: End 50-run loop ===

    %% --------------------------
    %% 4e. result (last run output shown)
    fprintf('Results for %s on %s (last run shown):\n', algorithm, dataset_name);
    fprintf('Accuracy: %.2f%%\n', accuracy);
    fprintf('Convergence speed (steps to near-optimum): %d\n', convergence_speed);
    fprintf('Search capability (best fitness): %.4f\n', search_capability);
    fprintf('Time taken (s): %.2fs\n\n', time_complexity);

    %% --------------------------
    %% 4f. Save to workspace & .mat
    dataset_varname = strrep(dataset_name, '.', '_');
    result_name = ['result_' algorithm '_' dataset_varname];
    assignin('base', result_name, struct( ...
        'accuracy', accuracy, ...
        'convergence_speed', convergence_speed, ...
        'search_capability', search_capability, ...
        'time_complexity', time_complexity, ...
        'Convergence_curve', Convergence_curve, ...
        'predictions', predictions));

    save([result_name '.mat'], 'accuracy', 'convergence_speed', 'search_capability', 'time_complexity', 'Convergence_curve', 'predictions');

    results(d).dataset = dataset_name;
    results(d).accuracy = accuracy;
    results(d).convergence_speed = convergence_speed;
    results(d).search_capability = search_capability;
    results(d).time_complexity = time_complexity;
end

%% ------------------------------
%% 5. Comparison table
fprintf('===============================\n');
fprintf('Comparison Table for %s\n', algorithm);
fprintf('===============================\n');
T = table({results.dataset}', [results.accuracy]', [results.convergence_speed]', ...
    [results.search_capability]', [results.time_complexity]', ...
    'VariableNames', {'Dataset','Accuracy','ConvergenceSpeed','SearchCapability','TimeComplexity'});
disp(T);

%% ------------------------------

%% ==============================
%  6. Summary statistics (50 runs per dataset)
%% ==============================

fprintf('\n===== SUMMARY RESULTS (50 RUNS EACH) =====\n');

summary_results = struct();

for d = 1:length(results)
    acc  = [results(d).runs.accuracy];
    time = [results(d).runs.time_complexity];
    sc   = [results(d).runs.search_capability];

    summary_results(d).dataset = results(d).dataset;

    % Accuracy stats
    summary_results(d).acc_min = min(acc);
    summary_results(d).acc_max = max(acc);
    summary_results(d).acc_mean = mean(acc);
    summary_results(d).acc_mode = mode(acc);
    summary_results(d).acc_median = median(acc);
    summary_results(d).acc_std = std(acc);

    % Time stats
    summary_results(d).time_min = min(time);
    summary_results(d).time_max = max(time);
    summary_results(d).time_mean = mean(time);
    summary_results(d).time_mode = mode(time);
    summary_results(d).time_median = median(time);
    summary_results(d).time_std = std(time);

    % Search capability stats
    summary_results(d).sc_min = min(sc);
    summary_results(d).sc_max = max(sc);
    summary_results(d).sc_mean = mean(sc);
    summary_results(d).sc_mode = mode(sc);
    summary_results(d).sc_median = median(sc);
    summary_results(d).sc_std = std(sc);

    % Display table for each dataset
    fprintf('\n--- %s Summary (%s) ---\n', algorithm, results(d).dataset);
    Tsum = table( ...
        summary_results(d).acc_min, summary_results(d).acc_max, summary_results(d).acc_mean, summary_results(d).acc_mode, summary_results(d).acc_median, summary_results(d).acc_std, ...
        summary_results(d).time_min, summary_results(d).time_max, summary_results(d).time_mean, summary_results(d).time_mode, summary_results(d).time_median, summary_results(d).time_std, ...
        summary_results(d).sc_min, summary_results(d).sc_max, summary_results(d).sc_mean, summary_results(d).sc_mode, summary_results(d).sc_median, summary_results(d).sc_std, ...
        'VariableNames', { ...
            'AccMin','AccMax','AccMean','AccMode','AccMedian','AccStd', ...
            'TimeMin','TimeMax','TimeMean','TimeMode','TimeMedian','TimeStd', ...
            'ScMin','ScMax','ScMean','ScMode','ScMedian','ScStd' });
    disp(Tsum);
end


%% ==============================
%  7. PLOTS (ALL GRAPHS)
%% ==============================

for d = 1:length(results)
    dataset_name = results(d).dataset;
    acc  = [results(d).runs.accuracy];
    time = [results(d).runs.time_complexity];
    sc   = [results(d).runs.search_capability];

    figure;
    boxplot(acc);
    title(['Accuracy Distribution (50 runs) - ' algorithm ' on ' dataset_name]);
    ylabel('Accuracy (%)');
    grid on;

    figure;
    boxplot(time);
    title(['Computational Time (50 runs) - ' algorithm ' on ' dataset_name]);
    ylabel('Time (seconds)');
    grid on;

    figure;
    boxplot(sc);
    title(['Search Capability (50 runs) - ' algorithm ' on ' dataset_name]);
    ylabel('Training Error (lower = better)');
    grid on;

    % Average convergence curve
    curves = cell2mat(arrayfun(@(x) x.Convergence_curve(:), results(d).runs, 'UniformOutput', false)');
    avg_curve = mean(curves, 1);

    figure;
    plot(avg_curve, 'LineWidth', 2);
    title(['Average Convergence Curve - ' algorithm ' on ' dataset_name]);
    xlabel('Iteration');
    ylabel('Fitness (Error)');
    grid on;
end

%% Global comparisons across datasets
acc_means = arrayfun(@(x) mean([x.runs.accuracy]), results);
time_means = arrayfun(@(x) mean([x.runs.time_complexity]), results);

figure;
bar(acc_means);
set(gca, 'XTickLabel', {results.dataset});
title(['Mean Accuracy Comparison (' algorithm ')']);
ylabel('Accuracy (%)');
grid on;

figure;
bar(time_means);
set(gca, 'XTickLabel', {results.dataset});
title(['Mean Computational Time Comparison (' algorithm ')']);
ylabel('Time (seconds)');
grid on;

%% Helper function: classification_error
function err = classification_error(w, X, y)
    % w = solution vector (feature selection)
    selected = w > 0.5;
    if sum(selected)==0
        err = length(y); 
        return;
    end
    X_sel = X(:, selected);
    mdl = fitcknn(X_sel, y, 'NumNeighbors', 5);
    pred = predict(mdl, X_sel);
    err = sum(pred ~= y);
end

