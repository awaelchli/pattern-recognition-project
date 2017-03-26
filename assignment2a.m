%% Pattern Recognition FS2017
%  SVM Test
clc;
clear;

%% Load the training and test set
fraction = 0.01;

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
[ bestC, bestGamma, acc ] = grid_search_svm(train_images, train_labels, options, Cs, gammas, K);
toc

%% Train again with selected hyperparameters
o = get_supported_options();
options = [o.kernel.rbf, o.quiet, o.cost(bestC), o.kernel.gamma(bestGamma)];
model = svmtrain(train_labels, train_images, options);

[~, test_accuracy, ~] = svmpredict(test_labels, test_images, model, []);