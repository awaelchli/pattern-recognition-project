function [ best_net, tr ] = mlp_cross_validation( data, labels, neurons, learning_rates )
% Performs cross-validation to optimize hyperparameters and returns the
% network with the best parameters:
%   - Number of neurons in hidden unit
%   - Learning rate


% Construct parameter grid
[ neuron_params, rate_params] = meshgrid(neurons, learning_rates);
n_params = numel(neuron_params);

% Use stochastic gradient descent
trainFcn = 'trainscg'; 

% Cross-Entropy is good for classification
performFcn = 'crossentropy';

% Stores each network and its performance
networks = cell(n_params, 2);
network_performances = zeros(n_params, 1);

for i = 1 : n_params
    
    learning_rate = rate_params(i);
    hiddenLayerSize = neuron_params(i);

    % Create a feedforward network
    net = patternnet(hiddenLayerSize, trainFcn, performFcn);
    net.trainParam.lr = learning_rate;
    
    % Stop training if the network fails to improve (validation checks)
    net.trainParam.max_fail = 6;
    
    % Divide data into training and validation set
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;

    net.plotFcns = {'plotperform', 'plotroc'};

    % Train the Network
    [net, tr] = train(net, data, labels);
    
    networks{i, 1} = net;
    networks{i, 2} = tr;
    network_performances(i) = tr.best_perf;
    
    
end

[~, ind] = min(network_performances);

best_net = networks{ind, 1};
tr = networks{ind, 2};


end

