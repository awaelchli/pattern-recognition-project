%%
clear all;
close all;

%% settings
c = 2.6;                    % learning rate
nbrOfHiddenNodes = 25;
nbrOfOutputNodes = 10;
nbrOfEpoches = 100;
lambda = 1;                 % regularization

%% Load the training set
fraction = 1;
train = csvread('../data/train.csv');
n_data = ceil(fraction * size(train, 1));
trainLabels = train(1 : n_data, 1);
trainData = train(1 : n_data, 2 : end);
clear train;

%% Randomly select 100 data points to display
sel = randperm(size(trainData, 1));
sel = sel(1:100);

displayData(trainData(sel, :));

%% Feature Selection/ Feature Descriptor
fprintf('\nTransform trainingset to hog\n');
for i=1:size(trainData,1)
    hg=hog(reshape(trainData(i,:),28,28),8);
    newData(i,:)= hg(:);
end
trainData=newData;

%% Load the test set
fraction = 1;
test = csvread('../data/test.csv');
n_data = ceil(fraction * size(test, 1));
testLabels = test(1 : n_data, 1);
testData = test(1 : n_data, 2 : end);
clear test;

%% Feature Selection/ Feature Descriptor
fprintf('\nTransform testset to hog\n');
clear newData;
for i=1:size(testData,1)
    hg=hog(reshape(testData(i,:),28,28),8);
    newData(i,:)= hg(:);
end
testData=newData;

%% train 
tic;
[Theta1, Theta2] = mlpTrain(trainLabels, trainData, c, nbrOfHiddenNodes, nbrOfOutputNodes, nbrOfEpoches, lambda);
toc;

%% classify
% training set accuracy
pred = mlpPredict(Theta1, Theta2, trainData);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == trainLabels)) * 100);

% test set accuracy
pred = mlpPredict(Theta1, Theta2, testData);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == testLabels)) * 100);


