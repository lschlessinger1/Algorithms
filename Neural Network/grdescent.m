function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
% function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
%
% INPUT:
% func function to minimize
% w0 = initial weight vector 
% stepsize = initial gradient descent stepsize 
% tolerance = if norm(gradient)<tolerance, it quits
%
% OUTPUTS:
% 
% w = final weight vector
%

if nargin<5,tolerance=1e-02;end;
w = w0;
prevLoss = 0;

for i = 1:maxiter
    [loss, gradient] = func(w);
    if norm(gradient) < tolerance
        break;
    end
    
    % Update learning rate
    % Increase the stepsize by a factor of 1.01 each iteration where the 
    % loss goes down, and decrease it by a factor 0.5 if the loss went up.
    % also undo the last update in that case to make sure the loss 
    % decreases every iteration
    if i > 1
        if loss - prevLoss < 0
            % loss decreased
            stepsize = stepsize * 1.01;
        else
            % loss increased
            stepsize = stepsize * 0.5;
        end
    end
    prevLoss = loss;
    
    % Update weights
    lambda = 0.03;
    w = w - stepsize * (gradient + lambda .* w);
end


