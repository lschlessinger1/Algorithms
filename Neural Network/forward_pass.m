function [ as,zs ] = forward_pass(W, xTr, trans_func )
% function [as,zs]=forward_pass(W,xTr,trans_func)
%
% INPUT:
% W weights (cell array)
% xTr dxn matrix (each column is an input vector)
% trans_func transition function to apply for inner layers
%
% OUTPUTS:
%
% as = result of forward pass 
% zs = result of forward pass (zs{1} output layer of the forward pass) 
%

n=size(xTr, 2);

% first, we add the constant weight
zs{length(W)+1}=[xTr;ones(1,n)];
as{length(W)+1}=xTr;
% Do the forward process here:
for i=length(W):-1:2
    as{i} = W{i} * zs{i+1};
%     as{i} = (as{i} - mean(as{i}, 2)) ./ std(as{i}, 0, 2); % batch norm
    zs{i} = [trans_func(as{i}); ones(1,n)];
end;
% (last one is special, no transition function)
as{1} = W{1} * zs{2};
zs{1} = as{1};
end

