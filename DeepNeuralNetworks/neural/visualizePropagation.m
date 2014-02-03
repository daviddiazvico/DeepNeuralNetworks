function [] = visualizePropagation(num, weights, data, activations, label, ...
                                   currentPath)
% Visualizes the learned weights, the original data and the propagated values.
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

L = length(weights);

% Plots the data
subplot(3, L, 1);
visualize(data(:, 1:16));
title('data');

for l = 1:L

    % Plots the weights
    subplot(3, L, L + l);
    visualize(cell2mat(weights(l)));
    title(strcat('learned weights.', num2str(l)));

    % Plots the representations
    subplot(3, L, 2*L + l);
    visualize(cell2mat(activations(l))(:, 1:16));
    title(strcat('representation.', num2str(l)));

end

% Saves the picture
print('-djpeg', strcat(currentPath, '/', label, '.jpg'));
