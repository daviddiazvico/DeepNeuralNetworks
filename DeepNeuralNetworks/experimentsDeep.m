% Deep neural networks experiments.
%
% Author:
%	David Diaz Vico

% Actualizes the path
paths;

% Load data
load mnist_classify_reduced;
data = data';
labels = labels';
testdata = testdata';
testlabels = testlabels';
[D,N] = size(data);

% Create targets
targets = labels2targets(labels);

% Define the deep network parameters
Mout = size(targets,1);
F = @logistic;
outF = @identity;

% Tries several numbers of hidden layers for each model
Larray = [1,2,3,4];
nlayers = length(Larray);

% Tries several hidden layer widths for each model
%M = 10.^2;
%M = 9.^2;
%M = 8.^2;
M = 7.^2;
%M = 6.^2;
%M = 5.^2;

% Trains the deep models
trainDeepModels;

% Tests the deep models
testDeepModels;

