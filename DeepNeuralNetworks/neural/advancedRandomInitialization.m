function [model] = advancedRandomInitialization(D,M)
% Returns intelligent random initial weights.
%
% Arguments:
%	D	the dimension of the input data
%	M	the number of units of the layers
%
% Returns:
%	model	the MLP model consisting of W and B
%
% Author:
%	David Diaz Vico

L = length(M);
W = cell(L);
B = cell(L);

% Calculates weights and biases for the input layer
P = sqrt(6)/sqrt(D+M(1));
W(1) = unifrnd(-P,P,D,M(1));
B(1) = unifrnd(-P,P,M(1),1);

% Calculates weights and biases for the other layers
for l = 2:L
	P = sqrt(6)/sqrt(M(l-1)+M(l));
	W(l) = unifrnd(-P,P,M(l-1),M(l));
	B(l) = unifrnd(-P,P,M(l),1);
end

model.W = W;
model.B = B;

