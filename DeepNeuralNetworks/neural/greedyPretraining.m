function [model] = greedyPretraining(X,M,F,outF,variant,varargin)
% Returns AE or RBM greedy-layer-wise-calculated initial weights.
%
% Arguments:
%	X		the training data
%	M		the number of units of the layers
%	F		the activation function in the hidden layers
%	dF		the derivative of the activation funcion in the hidden
%			layers
%	outF		the activation function in the output layer
%	doutF		the derivative of the activation function in the output
%			layer
%	variant		AE (to get a S_AE) or RBM (to get a DBN)
%	varargin	optional arguments for AE or RBM train functions
%
% Returns:
%	model	the MLP (S_AE or DBN) model consisting of W and B
%
% Author:
%	David Diaz Vico

[D,N] = size(X);
L = length(M);
W = cell(L);
B = cell(L);
Y = cell(L);

% Train and propagate through the first layer
if (isequal(variant,'AE'))
	auxmodel = advancedRandomInitialization(D,[M(1),D]);
	[auxmodel,errors] = mlpTrain(X,auxmodel,X,varargin{:});
elseif (isequal(variant,'RBM'))
	[auxmodel,errors] = rbmTrain(X,M(1),varargin{:});
end
W(1) = cell2mat(auxmodel.W(1));
B(1) = cell2mat(auxmodel.B(1));
Y(1) = cell2mat(propagate(auxmodel,F,outF,X)(1));

% Train and propagate through the other layers
for l = 2:L
	auxX = cell2mat(Y(l-1));
	if (isequal(variant,'AE'))
		auxmodel = advancedRandomInitialization(M(l-1),[M(l),M(l-1)]);
		[auxmodel,errors] = mlpTrain(auxX,auxmodel,auxX,varargin{:});
	elseif (isequal(variant,'RBM'))
		[auxmodel,errors] = rbmTrain(auxX,M(l),varargin{:});
	end
	W(l) = cell2mat(auxmodel.W(1));
	B(l) = cell2mat(auxmodel.B(1));
	Y(l) = cell2mat(propagate(auxmodel,F,outF,auxX)(1));
end

model.W = W;
model.B = B;

