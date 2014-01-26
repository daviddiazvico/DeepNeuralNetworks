function [y] = softsign(x)
% Softsign function.
%
% Arguments:
%	x	input
%
% Returns:
%	y	output
%
% Author:
%	David Diaz Vico

y = x./(1+abs(x));

