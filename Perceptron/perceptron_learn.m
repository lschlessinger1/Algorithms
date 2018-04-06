function [ w iterations ] = perceptron_learn( data_in )
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
    % y(t) ~= sign(w'(t) * x(t))
    [N, cols] = size(data_in);
    d = cols - 2;
    w = zeros(1, d + 1);
    iterations = 0;
    y = data_in(:, end);
    X = data_in(:, 1:end - 1);
    % y is ground truth label

    % while there exists at least 1 misclassification, run algorithm
    while sum(sign((w * X')') ~= y) > 0
        % get all misclassified examples
        misclassifiedExamples = find(sign((w * X')') ~= y);
        matrixSize = numel(misclassifiedExamples);
        % pick a random (x,y) that is misclassified
        randIndex = misclassifiedExamples(randperm(matrixSize, 1));
        xRandMisclassified = X(randIndex,:);
        yRandMisclassified = y(randIndex, end);
        % update w
        w = w + yRandMisclassified * xRandMisclassified;     
        % update iteration count
        iterations = iterations + 1;
    end
end

