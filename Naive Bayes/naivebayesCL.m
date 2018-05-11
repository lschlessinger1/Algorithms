function [w,b]=naivebayesCL(x,y);
% function [w,b]=naivebayesCL(x,y);
%
% Implementation of a Naive Bayes classifier
% Input:
% x : n input vectors of d dimensions (dxn)
% y : n labels (-1 or +1)
%
% Output:
% w : weight vector
% b : bias (scalar)
%

[d,n]=size(x);

% first get prior
[pOfYPos, pOfYNeg] = naivebayesPY(x, y);
% then get likelihood
[pOfXGivenYPos, pOfXGivenYNeg] = naivebayesPXY(x, y);

% if P(Y=1|x) >= P(Y=-1|x) then predict the label 1
% i.e. P(Y=1|x) / P(Y=-1|x >= 1
% taking the log, gives:
w = log(pOfXGivenYPos) - log(pOfXGivenYNeg);

b = log(pOfYPos) - log(pOfYNeg);