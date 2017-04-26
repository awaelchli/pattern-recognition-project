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

[Cs, Gs] = meshgrid(gridC, gridGamma);
accuracy = zeros(numel(Cs), 1);

%the parallels toolbox is needed to run parfor
parfor i = 1 : numel(Cs)
        
    C = Cs(i);
    gamma = Gs(i);
    fprintf("C = %.3d\tgamma = %.3d\t", C, gamma);
    current_options = [svm_options, opt.cost(C), opt.kernel.gamma(gamma), ...
                       opt.cross_validation(k)];
    accuracy(i) = svmtrain(train_labels, train_set, current_options);
end

% Select the parameters that performed the best
[max_acc, k] = max(accuracy);
C = Cs(k);
gamma = Gs(k);
accuracy = max_acc;

end

