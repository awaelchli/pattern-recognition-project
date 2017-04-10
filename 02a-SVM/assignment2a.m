%% Pattern Recognition FS2017
%  Exercise 2a
%  Group Pink
clc;
clear;

addpath('lib/libsvm-3.22/windows/');

%% Load the training and test set
[ train_images, train_labels, ...
  test_images, test_labels, shift, scale ] = load_MNIST('data/', 0.3);

%%  Train model using RBF kernel and grid search with cross validation
K = 10;
grid_resolution = 10;

Cs = 2 .^ linspace(-5, 15, grid_resolution);
gammas = 2 .^ linspace(-15, 3, grid_resolution);

o = get_supported_options();
options = [o.kernel.rbf, o.quiet, o.cachesize(8000)];

tic;
[ bestC, bestGamma, ~ ] = grid_search_svm(train_images, train_labels, options, Cs, gammas, K);
toc

% Train again with selected hyperparameters
o = get_supported_options();
options = [o.kernel.rbf, o.quiet, o.cachesize(8000), o.cost(bestC), o.kernel.gamma(bestGamma)];
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
options = [o.kernel.linear, o.quiet, o.cachesize(8000)];

tic;
[ bestC, ~, ~ ] = grid_search_svm(train_images, train_labels, options, Cs, 0, K);
toc

% Train again with selected hyperparameter C
o = get_supported_options();
options = [o.kernel.linear, o.quiet, o.cachesize(8000), o.cost(bestC)];
model = svmtrain(train_labels, train_images, options);

% Run svm on the unseen test set
[~, test_accuracy, ~] = svmpredict(test_labels, test_images, model, []);

fprintf('Accuracy on test set using linear kernel with C = %f: %f\n', bestC, test_accuracy(1));
