function K = computeK(kernel_type, X, Z, param)
% function K = computeK(kernel_type, X, Z)
% computes a matrix K such that Kij=g(x,z);
% for three different function linear, rbf or polynomial.
%
% Input:
% kernel_type: either 'linear','polynomial','rbf'
% X: n input vectors of dimension d (dxn);
% Z: m input vectors of dimension d (dxn);
% param: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)
%
% OUTPUT:
% K : nxm kernel matrix
%
%

if nargin<2,
	Z=X;
end;

if strcmp(kernel_type, 'linear')
    K = X' * Z;
elseif strcmp(kernel_type, 'rbf')
    gamma = param;
    K = exp(-gamma * (l2distance(X, Z) .^ 2));
elseif strcmp(kernel_type, 'poly')
    p = param;
    K = (X' * Z + 1) .^ p;
end
