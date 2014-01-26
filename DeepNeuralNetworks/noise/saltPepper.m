function [Y] = saltPepper(X,noisefactor)
% Adds salt and pepper noise to X.
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
Y = X;
corruption = binornd(1,noisefactor,D,N).*(-1+2*binornd(1,0.5,D,N));
Y(corruption==1) = 1;
Y(corruption==-1) = 0;

