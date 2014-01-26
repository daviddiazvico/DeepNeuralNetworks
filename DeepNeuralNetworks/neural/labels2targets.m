function [targets] = labels2targets(labels)
% Create targets from labels.
%
% Arguments:
%	labels	labels (natural numbers)
%
% Returns:
%	targets	targets (binary vectors)
%
% Author:
%	David Diaz Vico

[D,N] = size(labels);
U = unique(labels);
targets = zeros(nunique(labels),N);
for n = 1:length(U)
	targets(n,labels==U(n)) = 1;
end

