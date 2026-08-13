%% run_preprocess.m
clc; clear; close all;

datasets = {'kc1.csv', 'pc1.csv'};

for i = 1:length(datasets)
    [X_train, y_train, X_test, y_test] = preprocess_dataset(datasets{i});
    
    save(['train_test_' datasets{i} '.mat'], 'X_train', 'y_train', 'X_test', 'y_test');
end
