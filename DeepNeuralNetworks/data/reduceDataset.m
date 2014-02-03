function [] = reduceDataset(in, s, out)
% Reduces by a coefficient s the square images of a dataset.
%
% Arguments:
%	in	original dataset name
%	s	reduction coefficient
%	out	reduced dataset name
%
% Returns:
%
% Author:
%	David Diaz Vico

% Load data
load(in);

% Reduce data
[N, D] = size(data);
for n = 1:N
    reducedData(n, :) = reduceDim(data(n, :), s);
end;
[N, D] = size(testdata);
for n = 1:N
    reducedTestdata(n, :) = reduceDim(testdata(n, :), s);
end;

% Save data
data = reducedData;
testdata = reducedTestdata;
save(out, 'data', 'labels', 'testdata', 'testlabels');
