function [bestC,bestP,bestval,allvalerrs]=crossvalidate(xTr,yTr,ktype,Cs,paras)
% function [bestC,bestP,bestval,allvalerrs]=crossvalidate(xTr,yTr,ktype,Cs,paras)
%
% INPUT:	
% xTr : dxn input vectors
% yTr : 1xn input labels
% ktype : (linear, rbf, polynom ial)
% Cs   : interval of regularization constant that should be tried out
% paras: interval of kernel parameters that should be tried out
% 
% Output:
% bestC: best performing constant C
% bestP: best performing kernel parameter
% bestval: best performing validation error
% allvalerrs: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)
%
% Trains an SVM classifier for all possible parameter settings in Cs and paras and identifies the best setting on a
% validation split. 
%

bestC=0;
bestP=0;
bestval=10^10;

%% Split off validation data set
n = numel(yTr);
ii=randperm(n);
% shuffle first
x = xTr;
y = yTr;
xTr = xTr(:, ii);
yTr = yTr(ii);
nFolds = 4;
for q = 0:nFolds-1
    iiVal = (q *(n/nFolds) + 1):((q+1) * (n/nFolds));
    iiTr = setdiff(1:n, iiVal);
    
    xTv = xTr(:,iiVal);
    yTv = yTr(iiVal);
    
    xTr = xTr(:, iiTr);
    yTr =yTr(iiTr);
    %% Evaluate all parameter settings
    allvalerrs = inf * ones(numel(Cs), numel(paras));
    for i = 1:numel(Cs)
        for j = 1:numel(paras)
            C = Cs(i);
            kpar = paras(j);
            C=Cs(1)+rand()*(Cs(end)-Cs(1));
            kpar=paras(1)+rand()*(paras(end)-paras(1));
            
            [svmclassify, ~, ~]=trainsvm(xTr,yTr,C,ktype,kpar);
            valerr=sum(sign(svmclassify(xTv))~=yTv(:))/numel(yTv);
            
            allvalerrs(i, j) = valerr;
            if valerr<bestval
                bestval=valerr;
                bestP=kpar;
                bestC=C;
            end
        end
    end
    
    
    % reset xTr and yTr
    xTr = x(:, ii);
    yTr = y(ii);
end


