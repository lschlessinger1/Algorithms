function [ train_err, test_err ] = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use
    % keep track of N_train and N_test
    num_train_examples = size(X_tr, 1);
    num_test_examples = size(X_te, 1);
    
    % convert labels to {-1, +1}
    labels = unique(y_tr);
    neg_label = labels(1);
    pos_label = labels(2);
    y_tr(find(y_tr(:,1)==neg_label)) = -1;
    y_te(find(y_te(:,1)==neg_label)) = -1;
    y_tr(find(y_tr(:,1)==pos_label)) = 1;
    y_te(find(y_te(:,1)==pos_label)) = 1;
    
    % set uniform example weights
    D = ones(num_train_examples, 1) / num_train_examples;
    
    % initialize learner weights to 0
    alpha = zeros(n_trees, 1);
    
    % create hypotheses matrices for error bookkeeping
    tr_hypotheses = zeros(n_trees, num_train_examples);
    te_hypotheses = zeros(n_trees, num_test_examples);
    
    % keep track of error for plotting
    training_errors = zeros(n_trees, 1);
    test_errors = zeros(n_trees, 1);
    
    for i = 1:n_trees
        % train weak learner using distribution D
        stump = fitctree(X_tr, y_tr, 'SplitCriterion', 'deviance',...
            'MaxNumSplits', 1, 'Weights', D);
        
        % get weak hypothesis
        hypothesis = predict(stump, X_tr);
        tr_hypotheses(i, :) = hypothesis';
        
        % compute weighted error
        incorrect = y_tr ~= hypothesis;
        weighted_incorrect = D .* incorrect;
        error = sum(weighted_incorrect) / sum(D);
        
        % update learner i weights
        alpha(i) = (1/2) * log((1 - error) / error);
        
        % update weights
        Z_t = 2 * sqrt(error * (1 - error));
        new_weight = D .* exp(-alpha(i) * y_tr .* hypothesis);
        D = new_weight / Z_t;
       
        % calculate training error
        final_hypothesis = sign(sum(alpha(1:i) .* tr_hypotheses(1:i, :)))';
        train_err = sum(final_hypothesis ~= y_tr) / num_train_examples;
        training_errors(i, 1) = train_err;
       
        % calculate test error
        te_hypothesis = predict(stump, X_te);
        te_hypotheses(i, :) = te_hypothesis';
        final_te_hypothesis = sign(sum(alpha(1:i) .* te_hypotheses(1:i, :)))';
        test_err = sum(final_te_hypothesis ~= y_te) / num_test_examples;
        test_errors(i, 1) = test_err;
    end
    
end

