function [ C, gamma, accuracy ] = grid_search_svm( train_set, train_labels, svm_options, gridC, gridGamma, k)
%GRID_SEARCH_SVM Performs a grid search for SVM parameters C and \gamma
%
%   Inputs:     train_set:      The training set
%               train_labels:   Vector of labels for each training sample
%               svm_options:    Additional options to pass to the SVM 
%               gridC:          Grid-values for parameter C
%               gridGamma:      Grid-values for parameter \gamma
%               k:              Number of splits for k-fold
%                               cross-validation
%
%   Outputs:    C, gamma:       Parameters for the highest
%                               cross-validation accuracy
%               accuracy:       Best cross-validation accuracy


opt = get_supported_options();

grid_size = [numel(gridC), numel(gridGamma)];
accuracy = zeros(grid_size);

for i = 1 : grid_size(1)
    for j = 1 : grid_size(2)
        
        C = gridC(i);
        gamma = gridGamma(j);
        
        current_options = [svm_options, opt.cost(C), opt.kernel.gamma(gamma), ...
                           opt.cross_validation(k)];
        accuracy(i, j) = svmtrain(train_labels, train_set, current_options);
    end
end

[col_max, k] = max(accuracy, [], 1);
[max_acc, l] = max(col_max, [], 2);

C = gridC(k(l));
gamma = gridGamma(l);
accuracy = max_acc;

end

