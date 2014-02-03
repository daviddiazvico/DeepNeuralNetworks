function [] = visualizeReconstruction(num, weights, data, representations, ...
                                      reconstructions, label, currentPath)
% Visualizes the learned weights, the original data, the internal
% representations and the reconstructions.
%
% Arguments:
%	num		the window number
%	weights		the learned weights and biases
%	data		the data
%	representations	the internal representations of the data
%	reconstructions	the reconstructions of the data
%	label		window label
%
% Returns:
%
% Author:
%	David Diaz Vico

% Creates the plotting window
figure(num, 'Name', label);
title('label');

% Plots the weights
subplot(2, 2, 1);
visualize(weights);
title('learned weights');

% Plots the data
subplot(2, 2, 3);
visualize(data(:, 1:16));
title('data');

% Plots the representations
subplot(2, 2, 2);
visualize(representations(:, 1:16));
title('representation');

% Plots the reconstructions
subplot(2, 2, 4);
visualize(reconstructions(:, 1:16));
title('reconstruction');

% Saves the picture
print('-djpeg', strcat(currentPath, '/', label, '.jpg'));
