%% Pattern Recognition FS2017
%  Assignment 2b
%  Adrian Waelchli
clc;
clear;

%% Load the training and test set
fraction = 1;

train = csvread('data/train.csv');
n_train = ceil(fraction * size(train, 1));
train_labels = train(1 : n_train, 1)';
train_images = train(1 : n_train, 2 : end)';
clear train;

test = csvread('data/test.csv');
n_test = ceil(fraction * size(test, 1));
test_labels = test(1 : n_test, 1)';
test_images = test(1 : n_test, 2 : end)';
clear test;

%% Scale data
scale_factor = 1 / max(train_images(:));
train_images = train_images * scale_factor;
test_images = test_images * scale_factor;

%% Convert labels to logical vector

train_labels2 = zeros(10, n_train, 'logical');
inds = sub2ind(size(train_labels2), train_labels + 1, 1 : n_train);
train_labels2(inds) = 1;

test_labels2 = zeros(10, n_test, 'logical');
inds = sub2ind(size(test_labels2), test_labels + 1, 1 : n_test);
test_labels2(inds) = 1;

%% Train the network
% Perform cross-validation to optimize these hyperparameters:
% - Learning rate
% - Number of neurons in hidden layers

hidden_layer_sizes = 50;%10 : 20 : 100;
learning_rates = 0.01;%linspace(0.01, 1, 5);

tic;
[net, tr] = mlp_cross_validation(train_images, train_labels2, hidden_layer_sizes, learning_rates);
toc;

% These are the optimal hyper-parameters
learning_rate = net.trainParam.lr;
hidden_units = net.layers{1}.size();

% Show performance vs. iteration for the best network
fig = figure;
plotperform(tr);
perf_file = sprintf('Performance-vs-epochs-hidden=%d-LR=%f.png', hidden_units, learning_rate);
saveas(fig, perf_file);

% View the network that performed best
view(net);

fprintf('Optimal parameters: \n');
fprintf('\tLearning rate: %f \n', learning_rate);
fprintf('\tNumber of hidden units: %d \n', hidden_units);

%% Test the Network
% Feed the test images
% Output is a probability for every class
output = net(test_images);

% Classify by taking the class with the highest probability
[~, prediction] = max(output, [], 1);

% Convert class label to digit 
prediction = prediction - 1;

% Compute accuracy
accuracy = sum(prediction == test_labels) / n_test;

% Plot ROC curve
fig = figure;
plotroc(test_labels2, output);
roc_file = sprintf('ROC-hidden=%d-LR=%f.png', hidden_units, learning_rate);
saveas(fig, roc_file);

fprintf('Classification accuracy: %f\n', accuracy);
