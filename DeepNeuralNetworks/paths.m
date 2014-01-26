% Adds the subfolders to the path.
%
% Author:
%	David Diaz Vico

% Add paths
currentPath = strcat(pwd,'/');
addpath(currentPath);
addpath(strcat(currentPath,'activation/'));
addpath(strcat(currentPath,'data/'));
addpath(strcat(currentPath,'neural/'));
addpath(strcat(currentPath,'noise/'));
addpath(strcat(currentPath,'utilities/'));
addpath(strcat(currentPath,'experiments/'));

