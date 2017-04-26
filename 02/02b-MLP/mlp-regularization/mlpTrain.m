function [ Theta1, Theta2, J ] = mlpTrain( y, X, c, nbrOfHiddenNodes, nbrOfOutputNodes, nbrOfEpoches, lambda )
%MLP_TRAIN trains the multilayer perceptron
%   labels of the ground truth
%   X dataset each sample is a row vector
%   k number of hidden layers
%   c learning rate
%   w the generated weights
%   J error

    %% initialization of wheights
    
    fprintf('\nInitialize wheights... \n')
    y = y + 1; % add 1 to y, so that it can be used as index 0->1 ... 9 -> 10 remove after the prediction

    nbrOfInputNodes = size(X,2);
    initial_Theta1 = randInitializeWeights(nbrOfInputNodes, nbrOfHiddenNodes);
    initial_Theta2 = randInitializeWeights(nbrOfHiddenNodes, nbrOfOutputNodes);
    
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

    %% Train the network
    fprintf('\nTraining Neural Network... \n')
    options = optimset('MaxIter', nbrOfEpoches);

    costFunction = @(p) nnCostFunction(p, ...
                                       nbrOfInputNodes, ...
                                       nbrOfHiddenNodes, ...
                                       nbrOfOutputNodes, X, y, lambda, c);
                                   
                                   
    % optimize the costfunction with fmincg 
    [nn_params, J] = fmincg(costFunction, initial_nn_params, options);

    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:nbrOfHiddenNodes * (nbrOfInputNodes + 1)), ...
                     nbrOfHiddenNodes, (nbrOfInputNodes + 1));

    Theta2 = reshape(nn_params((1 + (nbrOfHiddenNodes * (nbrOfInputNodes + 1))):end), ...
                     nbrOfOutputNodes, (nbrOfHiddenNodes + 1));

end



