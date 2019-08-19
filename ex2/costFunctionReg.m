function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J_basic = (1 / m) * sum( -y .* log(sigmoid (X * theta)) - (1-y) .* log(1 - sigmoid(X * theta)) );

theta_reg = theta; 
theta_reg (1) = 0;

J_reg =  (lambda / (2 * m)) * sum(theta_reg .^ 2);

J = J_basic + J_reg;



% theta(0)
grad_temp_1  = (1/m * sum((sigmoid(X * theta) - y) .* X))'; 

%theta(1~)
grad_temp_2 = (1/m * sum((sigmoid(X * theta) - y) .* X))' + (lambda / m) .* theta ;  

n = length(theta);

%grad 
grad(1) = grad_temp_1(1);
grad(2:n) = grad_temp_2(2:n,:);




% =============================================================

end
