function [loss,gradient,preds]=hinge(w,xTr,yTr,lambda)
% function w=ridge(xTr,yTr,lambda)
%
% INPUT:
% xTr dxn matrix (each column is an input vector)
% yTr 1xn matrix (each entry is a label)
% lambda regression constant
% w weight vector (default w=0)
%
% OUTPUTS:
%
% loss = the total loss obtained with w on xTr and yTr
% gradient = the gradient at w
%

[d,n]=size(xTr);
h = xTr' * w;
loss = sum(max(ones(n, 1) - yTr' .* h, zeros(n, 1))) + lambda * (w' * w);

indices = (yTr' .* h) > 1;
yTr(indices) = 0;
xTr(:, indices) = 0;
gradient = sum(-xTr * yTr', 2) + 2 * lambda * w;
