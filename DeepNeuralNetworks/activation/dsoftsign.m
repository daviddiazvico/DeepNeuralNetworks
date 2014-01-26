function [y] = dsoftsign(x)
% Derivative of the softsign function with respect to x.
%
% Arguments:
%	x	input
%
% Returns:
%	y	output
%
% Author:
%	David Diaz Vico

y = 1./(1+abs(x)).^2;

