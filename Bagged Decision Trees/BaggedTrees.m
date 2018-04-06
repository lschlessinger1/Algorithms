function [ oobErr] = BaggedTrees( X, Y, numBags )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function
    num_samples = size(X, 1);
    
    oobErr = 0;
    oob_errs = zeros(numBags, 1);
    
    % create hypotheses matrices for error bookkeeping
    hypotheses = zeros(numBags, num_samples);
    trees = cell(numBags, 1);
    
    for i = 1:numBags
        % Create a bootstrap version of the training data
        [bag, indices] = datasample(X, num_samples);
        Y_bag = Y(indices');

        trees{i} = fitctree(bag, Y_bag);
            
        in_bag_indices = unique(indices);
        all_indices = 1:num_samples;
        
        oob_indices = setdiff(all_indices, in_bag_indices);
        
        X_test = X(oob_indices', :);
        hypothesis = predict(trees{i}, X_test);
        hypotheses(i, oob_indices') = hypothesis';
        
        % Use majority vote for prediction
        h_copy = hypotheses(1:i, :)';
        h_copy(h_copy == 0) = NaN;
        
        final_hypothesis = mode(h_copy, 2);
        final_hypothesis(isnan(final_hypothesis)) = 0;
        
        % Compute the OOB_i error for each response observation y_i in the 
        % training data and then compute average of OOB_i
        y_test = Y(final_hypothesis ~= 0);
        y_pred = final_hypothesis(final_hypothesis ~= 0);
        
        B = sum(final_hypothesis ~= 0);
        num_errors = sum(y_pred ~= y_test);
        
        oobErr = (1 / B) * num_errors;
        oob_errs(i) = oobErr;
    end
    
    plot(oob_errs, '--');
    hold on;
end
