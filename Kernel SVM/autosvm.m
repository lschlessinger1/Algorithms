function svmclassify=autosvm(xTr,yTr)
%	function svmclassify=autosvm(xTr,yTr)
% INPUT:	
% xTr : dxn input vectors
% yTr : 1xn input labels
% 
% Output:
% svmclassify : a classifier (scmclassify(xTe) returns the predictions on xTe)
%
%
% Performs cross validation to train an SVM with optimal hyper-parameters on xTr,yTr
%
% disp('Performing cross validation ...');
% [bestC,bestP]=crossvalidate(xTr,yTr,'rbf',2.^[-1:8],2.^[-2:3]);
[bestC,bestP,bestval]=crossvalidate(xTr,yTr,'rbf',2.^[-2:0.5:9],2.^[-3:0.5:4]);
% telescopic CV
[bestC,bestP,bestval]=crossvalidate(xTr,yTr,'rbf',bestC-10:2:bestC+10,bestP-1:0.25:bestP+1);
% disp('Training SVM ...');
svmclassify=trainsvm(xTr,yTr,bestC,'rbf',bestP);

