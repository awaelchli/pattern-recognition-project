%% Pattern Recognition FS2017
%  Assignment 2b
%  Adrian Waelchli
clc;
clear;

%% Load the training and test set
fraction = 0.1;

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
% TODO: Find optimal parameters

trainFcn = 'trainscg'; 

% Create a feedforward network
hiddenLayerSize = 50;
net = patternnet(hiddenLayerSize, trainFcn);

% Divide data into training and validation
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 40/100;

% Cross-Entropy is good for classification
net.performFcn = 'crossentropy';
net.plotFcns = {'plotperform', 'plotroc'};

% View the Network
view(net);

% Train the Network
[net, tr] = train(net, train_images, train_labels2);

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
figure, plotroc(test_labels2, output);

