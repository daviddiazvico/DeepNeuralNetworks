function [y] = tanhyp(x)
% Hyperbolic tangent function.
%
% Arguments:
%	x	input
%
% Returns:
%	y	output
%
% Author:
%	David Diaz Vico

a = exp(2*x);
y = (a-1)./(a+1);

