function [svmclassify,sv_i,alphas]=trainsvm(xTr,yTr, C,ktype,kpar);
% function [svmclassify,sv_i,alphas]=trainsvm(xTr,yTr, C,ktype,kpar);
% INPUT:	
% xTr : dxn input vectors
% yTr : 1xn input labels
% C   : regularization constant (in front of loss)
% ktype : (linear, rbf, polynomial)
% 
% Output:
% svmclassify : a classifier (scmclassify(xTe) returns the predictions on xTe)
% sv_i : indices of support vecdtors
% alphas : a nx1 vector of alpha values
%
% Trains an SVM classifier with kernel (ktype) and parameters (C,kpar)
% on the data set (xTr,yTr)
%

if nargin<5,kpar=1;end;
% yTr=yTr(:);
% svmclassify=@(xTe) (rand(1,size(xTe,2))>0.5).*2-1; %% classify everything randomly
n=length(yTr);



% disp('Generating Kernel ...')
% 
K = computeK(ktype, xTr, xTr, kpar);
%
% disp('Solving QP ...')
%
[H, q, Aeq, beq, lb, ub] = generateQP(K, yTr, C);
A = Aeq .* 0;
b = beq .* 0;
X0 = zeros(n, 1);
warning('off','all');
options = optimoptions(@quadprog,'Display','off');
alphas = quadprog(H, q, A, b, Aeq, beq, lb, ub, X0, options);
%
% disp('Recovering bias')
%
bias = recoverBias(K,yTr',alphas,C);
%
% disp('Extracting support vectors ...')
%
eps = 1e-6;
sv_i = find(abs(alphas - eps) < eps);
%
% disp('Creating classifier ...')
%
svmclassify = @(x) ((computeK(ktype, xTr, x, kpar)' * (alphas .* yTr')) + bias');
%
% disp('Computing training error:') % this is optional, but interesting to see
%
% trainerr=sum(sign(svmclassify(xTr)) ~= yTr(:)) / length(yTr)
%


