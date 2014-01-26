function [y] = dtanhyp(x)
% Derivative of the hyperbolic tangent function with respect to x.
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
y = 4*a./(a.^2+2*a+1);

