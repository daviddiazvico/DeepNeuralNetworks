% Shallow neural networks experiments.
%
% Author:
%	David Diaz Vico

% Actualizes the path
paths;

% Load data
load mnist_classify_reduced;
data = data';
testdata = testdata';
[D, N] = size(data);

% Tries several hidden layer widths for each model
Marray = [10.^2, 12.^2, 14.^2, 16.^2];
nwidths = length(Marray);

for Miter = 1:nwidths

    % Define the shallow network parameters
    M = [Marray(Miter), D];
    F = @logistic;
    dF = @dlogistic;
    outF = @identity;
    doutF = @didentity;

    % Initializes the shallow network
    auxmodel = advancedRandomInitialization(D, M);

    % Trains the AE and reconstructs the data, then visualizes the results
    [modelAE, errorsAE] = mlpTrain(data, auxmodel, data);
    activationAE = propagate(modelAE, F, outF, testdata);
    errorAE(Miter) = sum(sqrt(sum((cell2mat(activationAE(2)) - testdata) ...
                         .^2)))/N;
    visualizeReconstruction((Miter - 1)*5 + 1, cell2mat(modelAE.W(1)), ...
                            testdata, cell2mat(activationAE(1)), ...
                            cell2mat(activationAE(2)), strcat('AE ', ...
                            num2str(Marray(Miter))), strcat(currentPath, ...
                            '/experiments'));

    % Trains the DAE and reconstructs the data, then visualizes the results
    [modelDAE, errorsDAE] = mlpTrain(data, auxmodel, data, 'noiseF', @gaussian);
    activationDAE = propagate(modelDAE, F, outF, testdata);
    errorDAE(Miter) = sum(sqrt(sum((cell2mat(activationDAE(2)) - testdata) ...
                          .^2)))/N;
    visualizeReconstruction((Miter - 1)*5 + 2, cell2mat(modelDAE.W(1)), ...
                            testdata, cell2mat(activationDAE(1)), ...
                            cell2mat(activationDAE(2)), strcat('DAE ', ...
                            num2str(Marray(Miter))), strcat(currentPath, ...
                            '/experiments'));

    % Trains the SAE and reconstructs the data, then visualizes the results
    [modelSAE, errorsSAE] = mlpTrain(data, auxmodel, data, 'sparsity', true);
    activationSAE = propagate(modelSAE, F, outF, testdata);
    errorSAE(Miter) = sum(sqrt(sum((cell2mat(activationSAE(2)) - testdata) ...
                          .^2)))/N;
    visualizeReconstruction((Miter - 1)*5 + 3, cell2mat(modelSAE.W(1)), ...
                            testdata, cell2mat(activationSAE(1)), ...
                            cell2mat(activationSAE(2)), strcat('SAE ', ...
                            num2str(Marray(Miter))), strcat(currentPath, ...
                            '/experiments'));

    % Trains the SDAE and reconstructs the data, then visualizes the results
    [modelSDAE, errorsSDAE] = mlpTrain(data, auxmodel, data, 'sparsity', ...
                                       true, 'noiseF', @gaussian);
    activationSDAE = propagate(modelSDAE, F, outF, testdata);
    errorSDAE(Miter) = sum(sqrt(sum((cell2mat(activationSDAE(2)) - testdata ...
                           ).^2)))/N;
    visualizeReconstruction((Miter - 1)*5 + 4, cell2mat(modelSDAE.W(1)), ...
                            testdata, cell2mat(activationSDAE(1)), cell2mat( ...
                            activationSDAE(2)), strcat('SDAE ', num2str( ...
                            Marray(Miter))), strcat(currentPath, ...
                            '/experiments'));

    % Trains the RBM and reconstructs the data, then visualizes the results
    [modelRBM, errorsRBM] = rbmTrain(data, M(1));
    activationRBM = propagate(modelRBM, F, outF, testdata);
    errorRBM(Miter) = sum(sqrt(sum((cell2mat(activationRBM(2)) - testdata) ...
                          .^2)))/N;
    visualizeReconstruction((Miter - 1)*5 + 5, cell2mat(modelRBM.W(1)), ...
                            testdata, cell2mat(activationRBM(1)), cell2mat( ...
                            activationRBM(2)), strcat('RBM ', num2str( ...
                            Marray(Miter))), strcat(currentPath, ...
                            '/experiments'));

end

% Plots the mean reconstruction errors
figure((nwidths - 1)*5 + 6, 'Name', 'Reconstruction error');
plot(Marray, errorAE, Marray, errorDAE, Marray, errorSAE, Marray, errorSDAE);
legend('AE', 'DAE', 'SAE', 'SDAE');
xlabel('Number of hidden units');
ylabel('Mean reconstruction error');
title('Mean reconstruction error');
print('-djpeg', strcat(currentPath, '/experiments/ReconstructionError.jpg'));
