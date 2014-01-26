function [Y] = propagate(model,F,outF,data)
% Use MLP to propagate.
%
% Arguments:
%	model	the MLP model consisting of W and B
%	F		the activation function in the hidden layers
%	outF		the activation function in the output layer
%	data	input data for the MLP
%
% Returns:
%	Y	the testdata activations on each layer of the MLP
%
% Author:
%	David Diaz Vico

[D,N] = size(data);
[M,L] = size(model.W);
Y = cell(L);

if L > 1

	% Propagate through the first layer
	W = cell2mat(model.W(1));
	B = repmat(cell2mat(model.B(1)),1,N);
	Y(1) = F(W'*data+B);

	% Propagata through the hidden layers
	for l = 2:L-1
		W = cell2mat(model.W(l));
		X = cell2mat(Y(l-1));
		B = repmat(cell2mat(model.B(l)),1,N);
		Y(l) = F(W'*X+B);
	end

	% Propagate through the last layer
	W = cell2mat(model.W(L));
	X = cell2mat(Y(L-1));
	B = repmat(cell2mat(model.B(L)),1,N);
	Y(L) = outF(W'*X+B);

else

	% Propagate through the layer
	W = cell2mat(model.W(1));
	B = repmat(cell2mat(model.B(1)),1,N);
	Y(1) = outF(W'*data+B);

end

