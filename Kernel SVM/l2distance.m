function D=l2distance(X,Z)
% function D=l2distance(X,Z)
%	
% Computes the Euclidean distance matrix. 
% Syntax:
% D=l2distance(X,Z)
% Input:
% X: dxn data matrix with n vectors (columns) of dimensionality d
% Z: dxm data matrix with m vectors (columns) of dimensionality d
%
% Output:
% Matrix D of size nxm 
% D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
%
% call with only one input:
% l2distance(X)=l2distance(X,X)
%

[d,n]=size(X);
if (nargin==1) % case when there is only one input (X)
    D = l2distance(X, X);
else  % case when there are two inputs (X,Z)
    % bsxfun even faster
    [d,n] = size(X);
    [~,m] = size(Z);
    
%     G = X'*Z;
%     s = sum(X.^2, 1);
%     S = repmat(s', 1, m);
%     r = sum(Z.^2,1);
%     R = repmat(r,n,1);
%     D = S - 2 * G + R;

    D = repmat(sum(X.^2,1)',1,m) - 2*X'*Z + repmat(sum(Z.^2,1),n,1);
    % for numerical stability 
    D(D < 0) = 0;
    D = sqrt(D);
end;





