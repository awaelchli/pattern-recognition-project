function [ accuracy ] = train_svm( kernel, train_labels, train_data, test_labels, test_data, c, gamma, coef0, deg )

% cache size for libSVM in MB
cache_size = 1000;

switch kernel
    case kernels.linear
        kernel = 0;
        options = sprintf('-t %d -c %.10f -m %d -q', kernel, c, cache_size);
    case kernels.RBF
        kernel = 2;
        options = sprintf('-t %d -c %.10f -g %.10f -m %d -q', kernel, c, gamma, cache_size);
    case kernels.sigmoid
        kernel = 3;
        options = sprintf('-t %d -c %.10f -g %.10f -r %d -m %d -q', kernel, c, gamma, coef0, cache_size);
    otherwise % polynomial
        kernel = 1;
        options = sprintf('-t %d -c %.10f -g %.10f -r %d -m %d -d %d -q', kernel, c, gamma, coef0, cache_size, deg);
end

model = svmtrain(train_labels, train_data, options);
[~, accuracy, ~] = svmpredict(test_labels, test_data, model, '-q');

accuracy = accuracy(1);

end