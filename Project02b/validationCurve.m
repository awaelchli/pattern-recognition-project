function [error_train, error_val, error_test] = ...
    validationCurve(X, y, Xval, yval, Xtest, ytest, nbrOfOutputNodes, nbrOfHiddenNodes, lambdaList, c, nbrOfEpoches)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
% You need to return these variables correctly.
error_train = zeros(length(lambdaList), 1);
error_val = zeros(length(lambdaList), 1);
error_test = zeros(length(lambdaList), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%
i = 0;
for lambda = lambdaList
    i=i+1;
    [Theta1, Theta2, Jtrain] = mlpTrain(y, X, c, nbrOfHiddenNodes, nbrOfOutputNodes, nbrOfEpoches, lambda);
    [Jcross, GradCross] = nnCostFunction([Theta1(:);Theta2(:)], ...
                                   nbrOfInputNodes, ...
                                   nbrOfHiddenNodes, ...
                                   nbrOfOutputNodes, ...
                                   Xval, yval, lambda, c);
    [Jtest, GradCross] = nnCostFunction([Theta1(:);Theta2(:)], ...
                                   nbrOfInputNodes, ...
                                   nbrOfHiddenNodes, ...
                                   nbrOfOutputNodes, ...
                                   Xtest, ytest, lambda, c);
    
    error_train(i) = Jtrain(end);
    error_val(i) = Jcross(end);
    error_test(i) = Jtest(end);
end

% =========================================================================

end
