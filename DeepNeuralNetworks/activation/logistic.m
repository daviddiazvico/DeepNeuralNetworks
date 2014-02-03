function [y] = logistic(x)
% Sigmoid function.
%
% Arguments:
%	x	input
%
% Returns:
%	y	output
%
% Author:
%	David Diaz Vico

y = 1./(1 + exp(-x));
