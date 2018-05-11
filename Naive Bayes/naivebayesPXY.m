function [posprob,negprob] = naivebayesPXY(x,y)
% function [posprob,negprob] = naivebayesPXY(x,y);
%
% Computation of P(X|Y)
% Input:
% x : n input vectors of d dimensions (dxn)
% y : n labels (-1 or +1) (1xn)
%
% Output:
% posprob: probability vector of p(x|y=1) (dx1)
% negprob: probability vector of p(x|y=-1) (dx1)
%

% add one all-ones positive and negative example
[d,n]=size(x);
x=[x ones(d,2)];
y=[y -1 1];

[d,n] = size(x);

% smoothing parameter (Laplace smoothing)
l = 0;

% multinomial distribution likelihood
numPosOccurs = (sum(x .* sign(y + 1), 2) + l);
lengthOfPos = sum((sign(y + 1) .* sum(x)) + d * l);
posprob =  numPosOccurs ./ lengthOfPos;

numNegOccurs = (abs(sum(x .* sign(y - 1), 2) + l));
lengthOfNeg = abs(sum((sign(y - 1) .* sum(x))) + d * l);
negprob = numNegOccurs ./ lengthOfNeg;
