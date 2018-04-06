function [ w, e_in ] = logistic_reg( X, y, w_init, max_its, eta )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix (without an initial column of 1s)
%       y : data labels (plus or minus 1)
%       w_init: initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta: learning rate
    
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)

    w = w_init;
    N = size(X, 1);
    
%     prepend a column of ones to X
    X = [ones(size(X, 1), 1) X];
    % 10e-3 for first part, 10e-6 for last part
    tolerance = 10e-3;
    
    for t = 1:max_its
        pointwise_errors = (y .* X) ./ (1 + exp(y .* X * w));
        gradient = ((-1 / N) * sum(pointwise_errors))';
        if all(abs(gradient(:)) < tolerance)
            % if magnitude of all terms in gradient < tolerance, stop
            fprintf('terminating early at iteration %d \n', t);
            break
        end
        % update weights
        w = w - eta * gradient;
    end
    
    e_in = (1 / N) * sum(log(1 + exp(-y .* X * w)));
end

