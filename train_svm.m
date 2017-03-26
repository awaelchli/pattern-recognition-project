function [ accuracy ] = train_svm( train_labels, train_data, test_labels, test_data, c, gamma )

% cache size for libSVM in MB
cache_size = 1000;

% Linear kernel = 1, RBF kernel = 2
if nargin < 6
    % no gamma provided, use linear kernel
    kernel = 1;
    options = sprintf('-t %d -c %.10f -m %d -q', kernel, c, cache_size);
else
    % use RBF kernel
    kernel = 2;
    options = sprintf('-t %d -c %.10f -g %.10f -m %d -q', kernel, c, gamma, cache_size);
end

model = svmtrain(train_labels, train_data, options);
[~, accuracy, ~] = svmpredict(test_labels, test_data, model, '-q');

accuracy = accuracy(1);

end

