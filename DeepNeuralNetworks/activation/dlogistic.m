function [y] = dlogistic(x)
% Derivative of the sigmoid function with respect to x.
%
% Arguments:
%	x	input
%
% Returns:
%	y	output
%
% Author:
%	David Diaz Vico

y = exp(-x)./(1 + exp(-x)).^2;
