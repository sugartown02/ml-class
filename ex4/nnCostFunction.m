function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
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

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%num_label; 
y = eye(num_labels)(y,:);

% inputlayer
a_1 = [ones(m,1) X];

% hiddenlayer
a_2 = sigmoid(a_1 * Theta1');
a_2 = [ones(m,1) a_2] ;

% outputlayer
a_3 = sigmoid(a_2 * Theta2');

% cost
J = (1/m) * sum(sum((-y .* log(a_3)) - ((1-y) .* log(1-a_3))));

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));
% Dvec = [Theta1_grad(:); Theta2_grad(:)];

%세타제로 지우기 
%Theta_backprop_2 = Theta2(:,2:end);
%a2의 1항 지우기
%a_backprop_2 = a_2(:,2:end);


%델타3 구하기 
delta3 = a_3 - y; % 5000*10
%델타2 구하기 5000*26 = 5000*10 *10*26/ 10*26/10*26;  
delta2 = (delta3 * Theta2) .* (a_2 .* (1 - a_2)); 


%삼각형D구하기
D1 = (delta2(:,2:end))' * a_1; 
D2 = delta3' * a_2;

%그레디언트 구하기 
Theta1_grad = (1 / m) * D1 ;
Theta2_grad = (1 / m) * D2 ;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% regularization
Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;
Theta2_reg = Theta2;
Theta2_reg(:,1) = 0; 

J_reg = (lambda / (2 * m)) * (sum(sum(Theta1_reg .^ 2)) ...
	     + sum(sum(Theta2_reg .^ 2)));

J = J + J_reg;


%그래이드 구하기 
Theta1_grad_reg = (lambda/m) * Theta1_reg ;
Theta2_grad_reg = (lambda/m) * Theta2_reg ;

Theta1_grad = Theta1_grad + Theta1_grad_reg;
Theta2_grad = Theta2_grad + Theta2_grad_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
