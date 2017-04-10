function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda, c)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda,c) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m,1),X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1),a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

htheta = a3;
yVect = zeros(m,size(Theta2,1));
sampleVect = zeros(m,1);
for i = 1:m
    yVect(i,y(i)) = 1;
    sampleVect(i) = -yVect(i,:)*log(htheta(i,:))'-(1-yVect(i,:))*log(1-htheta(i,:))';
end

J = 1/m*sum(sampleVect);

% --- regularization ---
J = J + lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

% --- gradient ---
for i = 1:m
    % calculate forward prop
    a1 = [1,X(i,:)];
    z2 = a1*Theta1';
    a2 = sigmoid(z2);
    a2 = [1,a2];
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    % calculate backprop
    delta3 = a3 - yVect(i,:);
    thdelta2 = Theta2'*delta3';
    thdelta2 = thdelta2(2:end);
    delta2 = thdelta2'.*sigmoidGradient(z2);
    
    % input + output layer = 2
    Theta1_grad = Theta1_grad + c*delta2'*a1;
    Theta2_grad = Theta2_grad + c*delta3'*a2;
end

% calculate the gradient
Theta2_grad = 1/m*Theta2_grad;
Theta1_grad = 1/m*Theta1_grad;

% regularize the gradient
Theta2(:,1) = zeros(size(Theta2,1),1);
Theta1(:,1) = zeros(size(Theta1,1),1);
Theta2_grad = Theta2_grad + lambda/m*Theta2;
Theta1_grad = Theta1_grad + lambda/m*Theta1;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
