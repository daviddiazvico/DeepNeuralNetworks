function [y] = reduceDim(x, s)
% Reduces a square image by a coefficient s.
%
% Arguments:
%	x	input
%	s	reduction coefficient
%
% Returns:
%	y	output
%
% Author:
%	David Diaz Vico

% Transforms the image vector to a square matrix
d1 = floor(sqrt(length(x)));
for i = 1:d1
    for j = 1:d1
        A(i, j) = x(d1*(i - 1) + j);
    end
end

% Pics submatrices as the new pixels
d2 = floor(d1/s);
for i = 1:d2
    for j = 1:d2
        y(d2*(i - 1) + j) = sum(sum(A(floor(s*(i - 1) + 1):floor(s*i), ...
                                floor(s*(j - 1) + 1):floor(s*j))));
    end
end
M = max(1, max(y));
y = y/M;
