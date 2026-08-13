function [X_train, y_train, X_test, y_test] = preprocess()
    % PREPROCESS: load and prepare dataset for training/testing

    % === 1. Choose which dataset to use ===
    filename = 'pc1.csv';   % or change to 'kc1.csv'
    data = readtable(filename);

    % === 2. Handle column names and missing data ===
    data = rmmissing(data);
    warning('off','MATLAB:table:ModifiedAndSavedVarnames');

    % === 3. Split features and labels ===
    X = table2array(data(:, 1:end-1));   % all columns except last
    y = table2array(data(:, end));       % last column is label

    % === 4. Split into training (70%) and testing (30%) ===
    n = size(X,1);
    idx = randperm(n);
    n_train = round(0.7 * n);

    X_train = X(idx(1:n_train), :);
    y_train = y(idx(1:n_train), :);
    X_test  = X(idx(n_train+1:end), :);
    y_test  = y(idx(n_train+1:end), :);

    % === 5. Display summary ===
    fprintf('? Preprocessing done!\n');
    fprintf('Training data: %d samples\n', size(X_train,1));
    fprintf('Testing data: %d samples\n', size(X_test,1));
end

