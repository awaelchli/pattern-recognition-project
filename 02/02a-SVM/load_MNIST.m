function [ trainI, trainL, testI, testL, shift, scale ] = load_MNIST( folder, fraction )
% LOAD_MNIST Loads the MNIST training and test data and applies
% normalization

train = csvread(fullfile(folder, 'train.csv'));
n_train = ceil(fraction * size(train, 1));
trainL = train(1 : n_train, 1);
trainI = train(1 : n_train, 2 : end);

test = csvread(fullfile(folder, 'test.csv'));
n_test = ceil(fraction * size(test, 1));
testL = test(1 : n_test, 1);
testI = test(1 : n_test, 2 : end);

shift = min(trainI(:));
scale = 1 / max(trainI(:));

% Apply normalization
trainI = (trainI - shift) * scale;
testI = (testI - shift) * scale;

end

