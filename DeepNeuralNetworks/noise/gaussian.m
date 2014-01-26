function [Y] = gaussian(X,noisefactor)
% Adds gaussian noise to X.
%
% Arguments:
%	X	input
%
% Returns:
%	Y	output
%
% Author:
%	David Diaz Vico

[D,N] = size(X);
Y = X+normrnd(0,noisefactor,D,N);

