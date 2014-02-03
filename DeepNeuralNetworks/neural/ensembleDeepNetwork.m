function [model] = ensembleDeepNetwork(hiddenModel, outputModel)
% Assembles a deep neural network.
%
% Arguments:
%	hiddenModel	the MLP model of the hidden layers consisting of W and B
%	outputModel	the MLP model of the output layers consisting of W and B
%
% Returns:
%	model	the MLP model of the complete deep netowrk consisting of W and B
%
% Author:
%	David Diaz Vico

L1 = length(hiddenModel.W);
L2 = length(outputModel.W);
W = cell(1, L1 + L2);
B = cell(1, L1 + L2);

% Copies the hidden layers
for l = 1:L1
    W(l) = cell2mat(hiddenModel.W(l));
    B(l) = cell2mat(hiddenModel.B(l));
end

% Copies the output layers
for l = 1:L2
    W(L1 + l) = cell2mat(outputModel.W(l));
    B(L1 + l) = cell2mat(outputModel.B(l));
end

model.W = W;
model.B = B;
