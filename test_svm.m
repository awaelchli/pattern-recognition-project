%% Pattern Recognition FS2017
clc;

addpath('lib/libsvm-3.22/windows');

%% Experiment parameters
verbose = true;
fraction = 0.01;

% number of values for the training grid search
grid_search_size = 8;

%% Experiment
fprintf('Started experiment.\n');

%% Load data
fprintf('Loading data.\n');
if ~exist('train_images', 'var') || ~exist('test_images', 'var') || used_fraction ~= fraction
    
    used_fraction = fraction;
    
    % Load the training set
    train = csvread('data/train.csv');
    n_data = ceil(fraction * size(train, 1));
    train_labels = train(1 : n_data, 1);
    train_images = train(1 : n_data, 2 : end);
    
    clear train;
    
    % Load the test set
    test = csvread('data/test.csv');
    n_data = ceil(fraction * size(test, 1));
    test_labels = test(1 : n_data, 1);
    test_images = test(1 : n_data, 2 : end);
    clear test;
    
    % scale data
    max_value = 255;
    min_value = 0;
    diff = max_value - min_value;
    train_images = (train_images - min_value) ./ diff;
    test_images = (test_images - min_value) ./ diff;
end

%% Setup grid search parameters
% see http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf, page 5
c_ks = linspace(-2, 15, grid_search_size);
gamma_ks = linspace(-15, 1, grid_search_size);

%% Training with linear and RBF kernels (grid search for parameters)
fprintf('Training multi-class SVM.\n');
tic;

max_linear_accuracy = 0;
best_linear_c = 0;

max_rbf_accuracy = 0;
best_rbf_c = 0;
best_gamma = 0;

for m=1:grid_search_size
    c = 2^c_ks(m);
    
    % perform training with linear kernel
    linear_accuracy = train_svm(train_labels, train_images, test_labels, test_images, c);
    if verbose
        fprintf('Linear kernel, C = %.5f, gamma = %.4f. Accuracy: %.4f\n', c, gamma, linear_accuracy);
    end
    
    if linear_accuracy > max_linear_accuracy
        max_linear_accuracy = linear_accuracy;
        best_linear_c = c;
    end
    
    for n=1:grid_search_size
        gamma = 2^gamma_ks(n);
        
        % perform training with RBF kernel
        rbf_accuracy = train_svm(train_labels, train_images, test_labels, test_images, c, gamma);
        
        if verbose
            fprintf('RBF kernel, C = %.5f, gamma = %.4f. Accuracy: %.4f\n', c, gamma, rbf_accuracy);
        end
        
        if rbf_accuracy > max_rbf_accuracy
            max_rbf_accuracy = rbf_accuracy;
            best_rbf_c = c;
            best_gamma = gamma;
        end
    end
end

toc;

fprintf('\nResults for linear kernel:\n');
fprintf('Best accuracy: %.4f\n', max_linear_accuracy);
fprintf('Best parameters: C = %.5f.\n', best_linear_c);

fprintf('\nResults for RBF kernel:\n');
fprintf('Best accuracy: %.4f\n', max_rbf_accuracy);
fprintf('Best parameters: C = %.5f, gamma = %.5f.\n', best_rbf_c, best_gamma);