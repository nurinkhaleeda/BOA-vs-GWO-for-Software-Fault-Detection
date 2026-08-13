function [X_train, y_train, X_test, y_test] = preprocess_dataset(dataset_file, test_ratio)
    % Default test_ratio = 0.3 if not provided
    if nargin < 2
        test_ratio = 0.3;
    end

    %% 1. Load dataset
    fprintf('Loading %s ...\n', dataset_file);
    T = readtable(dataset_file);  % Automatically handles headers

    %% 2. Handle missing data
    % Replace missing numeric values with column mean
    numericVars = varfun(@isnumeric, T, 'OutputFormat', 'uniform');
    for i = find(numericVars)
        col = T{:, i};
        if any(ismissing(col))
            col(ismissing(col)) = mean(col(~ismissing(col)));
            T{:, i} = col;
        end
    end

    %% 3. Separate features and labels
    X = T{:,1:end-1};  % All columns except last
    y = T{:,end};      % Last column as label

    %% 4. Normalization (Min-Max 0-1)
    X = (X - min(X)) ./ (max(X) - min(X));

    %% 5. Split train and test
    N = size(X,1);
    idx = randperm(N);
    n_test = round(N * test_ratio);
    test_idx = idx(1:n_test);
    train_idx = idx(n_test+1:end);

    X_train = X(train_idx, :);
    y_train = y(train_idx, :);
    X_test = X(test_idx, :);
    y_test = y(test_idx, :);

    %% 6. Display summary
    fprintf('Dataset Summary for %s:\n', dataset_file);
    fprintf('Total samples: %d\n', N);
    fprintf('Features: %d\n', size(X,2));
    fprintf('Training samples: %d\n', size(X_train,1));
    fprintf('Testing samples: %d\n', size(X_test,1));
    fprintf('Missing values handled: Yes\n');
    fprintf('Normalization applied: Min-Max [0,1]\n\n');
end