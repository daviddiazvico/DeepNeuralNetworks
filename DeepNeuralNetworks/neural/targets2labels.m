function [labels] = targets2labels(targets)
% Create labels from targets.
%
% Arguments:
%	targets	targets (binary vectors)
%
% Returns:
%	labels	labels (natural numbers)
%
% Author:
%	David Diaz Vico

[D,N] = size(targets);
for n = 1:N
	[predictions(1,n),labels(1,n)] = max(targets(:,n));
end

