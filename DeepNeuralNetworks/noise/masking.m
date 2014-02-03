function [Y] = masking(X, noisefactor)
% Adds masking noise to X.
%
% Arguments:
%	X	input
%
% Returns:
%	Y	output
%
% Author:
%	David Diaz Vico

[D, N] = size(X);
Y = X;
corruption = binornd(1, noisefactor, D, N);
Y(corruption == 1) = 0;
