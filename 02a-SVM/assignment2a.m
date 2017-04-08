%% Pattern Recognition FS2017
%  Exercise 2a
%  Adrian Waelchli
clc;
clear;

%% Load the training and test set
fraction = 0.02;

train = csvread('data/train.csv');
n_train = ceil(fraction * size(train, 1));
train_labels = train(1 : n_train, 1);
train_images = train(1 : n_train, 2 : end);
clear train;

test = csvread('data/test.csv');
n_test = ceil(fraction * size(test, 1));
test_labels = test(1 : n_test, 1);
test_images = test(1 : n_test, 2 : end);
clear test;

%% Scale data
scale_factor = 1 / max(train_images(:));
train_images = train_images * scale_factor;
test_images = test_images * scale_factor;

%%  Train model using RBF kernel and grid search with cross validation
K = 10;
grid_resolution = 10;

Cs = 2 .^ linspace(-5, 15, grid_resolution);
gammas = 2 .^ linspace(-15, 3, grid_resolution);

o = get_supported_options();
options = [o.kernel.rbf, o.quiet];

tic;
[ bestC, bestGamma, ~ ] = grid_search_svm(train_images, train_labels, options, Cs, gammas, K);
toc

% Train again with selected hyperparameters
o = get_supported_options();
options = [o.kernel.rbf, o.quiet, o.cost(bestC), o.kernel.gamma(bestGamma)];
model = svmtrain(train_labels, train_images, options);

% Run svm on the unseen test set
[~, test_accuracy, ~] = svmpredict(test_labels, test_images, model, []);

fprintf('Accuracy on test set using RBF kernel with C = %f and gamma = %f: %f\n', bestC, bestGamma, test_accuracy(1));

%% Train model using LINEAR kernel and grid search with cross validation
% Same as above but the linear kernel has no parameter gamma. Perform grid
% search in 1D on cost C

K = 10;
grid_resolution = 10;

Cs = 2 .^ linspace(-5, 15, grid_resolution);

o = get_supported_options();
options = [o.kernel.linear, o.quiet];

tic;
[ bestC, ~, ~ ] = grid_search_svm(train_images, train_labels, options, Cs, 0, K);
toc

% Train again with selected hyperparameter C
o = get_supported_options();
options = [o.kernel.linear, o.quiet, o.cost(bestC)];
model = svmtrain(train_labels, train_images, options);

% Run svm on the unseen test set
[~, test_accuracy, ~] = svmpredict(test_labels, test_images, model, []);

fprintf('Accuracy on test set using linear kernel with C = %f: %f\n', bestC, test_accuracy(1));
