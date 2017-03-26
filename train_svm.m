%% Pattern Recognition FS2017
clc;

addpath('lib/libsvm-3.22/windows');

%% Experiment parameters
verbose = true;
fraction = 0.01;

% number of values for the training grid search
grid_search_size = 8;
coef_search_size = 11;
deg_search_size = 4;

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
coef0_ks = linspace(-5, 5, coef_search_size);
deg_ks = linspace(2, 5, deg_search_size);

%% Training with linear and RBF kernels (grid search for parameters)
fprintf('Training multi-class SVM.\n');
tic;

max_linear_accuracy = 0;
best_linear_c = 0;

max_rbf_accuracy = 0;
best_rbf_c = 0;
best_rbf_gamma = 0;

max_sigmoid_accuracy = 0;
best_sigmoid_c = 0;
best_sigmoid_gamma = 0;
best_sigmoid_coef0 = 0;

max_polynomial_accuracy = 0;
best_polynomial_c = 0;
best_polynomial_gamma = 0;
best_polynomial_coef0 = 0;
best_deg = 0;

for m=1:grid_search_size
    c = 2^c_ks(m);
    
    % perform training with linear kernel
    linear_accuracy = test_svm(kernels.linear, train_labels, train_images, test_labels, test_images, c);
    if verbose
        fprintf('Linear kernel, C = %.5f, gamma = %.5f. Accuracy: %.4f\n', c, gamma, linear_accuracy);
    end
    
    if linear_accuracy > max_linear_accuracy
        max_linear_accuracy = linear_accuracy;
        best_linear_c = c;
    end
    
    for n=1:grid_search_size
        gamma = 2^gamma_ks(n);
        
        % perform training with RBF kernel
        rbf_accuracy = test_svm(kernels.RBF, train_labels, train_images, test_labels, test_images, c, gamma);
        
        if verbose
            fprintf('RBF kernel, C = %.5f, gamma = %.5f. Accuracy: %.4f\n', c, gamma, rbf_accuracy);
        end
        
        if rbf_accuracy > max_rbf_accuracy
            max_rbf_accuracy = rbf_accuracy;
            best_rbf_c = c;
            best_rbf_gamma = gamma;
        end
        
        % LL: perform sigmoid kernel training
        for p=1:coef_search_size  
            coef0 = coef0_ks(p);
            sigmoid_accuracy = test_svm(kernels.sigmoid, train_labels, train_images, test_labels, test_images, c, gamma, coef0);
        
            if verbose
                fprintf('sigmoid kernel, C = %.5f, gamma = %.5f, coef0 = %d. Accuracy: %.4f\n', c, gamma, coef0, sigmoid_accuracy);
            end
        
            if sigmoid_accuracy > max_sigmoid_accuracy
                max_sigmoid_accuracy = sigmoid_accuracy;
                best_sigmoid_c = c;
                best_sigmoid_gamma = gamma;
                best_sigmoid_coef0 = coef0;
            end
            
            %LL: perform polynomial kernel training
            for q=1:deg_search_size
                deg = deg_ks(q);
                polynomial_accuracy = test_svm(kernels.polynomial, train_labels, train_images, test_labels, test_images, c, gamma, coef0, deg);
                
                if verbose
                    fprintf('polynomial kernel, C = %.5f, gamma = %.5f, coef0 = %d, degree = %d. Accuracy: %.4f\n', c, gamma, coef0, deg, sigmoid_accuracy);
                end
                
                if polynomial_accuracy > max_polynomial_accuracy
                    fprintf("NOW!!!");
                    max_polynomial_accuracy = polynomial_accuracy;
                    best_polynomial_c = c;
                    best_polynomial_gamma = gamma;
                    best_polynomial_coef0 = coef0;
                    best_deg = deg;
                    fprintf("max_polynomial_accuracy = %.4f, best_polynomial_c = %.5f, best_polynomial_gamma = %.5f, best_polynomial_coef0 = %d, best_deg = %d\n", max_polynomial_accuracy, best_polynomial_c, best_polynomial_gamma, best_polynomial_coef0, best_deg);
                end
            end
        end
    end
end

toc;

fprintf('\nResults for linear kernel:\n');
fprintf('Best accuracy: %.4f\n', max_linear_accuracy);
fprintf('Best parameters: C = %.5f.\n', best_linear_c);

fprintf('\nResults for RBF kernel:\n');
fprintf('Best accuracy: %.4f\n', max_rbf_accuracy);
fprintf('Best parameters: C = %.5f, gamma = %.5f.\n', best_rbf_c, best_rbf_gamma);

fprintf('\nResults for sigmoid kernel:\n');
fprintf('Best accuracy: %.4f\n', max_sigmoid_accuracy);
fprintf('Best parameters: C = %.5f, gamma = %.5f, coef0 = %d.\n', best_rbf_c, best_sigmoid_gamma, best_sigmoid_coef0);

fprintf('\nResults for polynomial kernel:\n');
fprintf('Best accuracy: %.4f\n', max_polynomial_accuracy);
fprintf('Best parameters: C = %.5f, gamma = %.5f, coef0 = %d, deg = %d.\n', best_polynomial_c, best_polynomial_gamma, best_polynomial_coef0, best_deg);
