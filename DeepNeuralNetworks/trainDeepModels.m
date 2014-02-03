% Deep neural networks training.
%
% Author:
%	David Diaz Vico

% Ensembles the PCP
perceptron = advancedRandomInitialization(D, [M, Mout]);
[model, err] = mlpTrain(data, perceptron, targets);
save(strcat(currentPath, '/experiments/PCP', num2str(1), '.', num2str(M), ...
     '.mat'), 'model');

% Intialize the deep network final classifier
classifierM = [M, Mout];
classifierL = length(classifierM);
modelClassifier = advancedRandomInitialization(M, classifierM);
hiddenL = Larray(nlayers);
hiddenM = repmat(M, 1, hiddenL);

% Ensembles the MLP
auxModel1 = randomInitialization(D, hiddenM);
for l = nlayers:-1:1
    auxModel1.W = auxModel1.W(1:l);
    auxModel1.B = auxModel1.B(1:l);
    auxModel2 = ensembleDeepNetwork(auxModel1, modelClassifier);
    [model, err] = mlpTrain(data, auxModel2, targets);
    save(strcat(currentPath, '/experiments/MLP', num2str(l), '.', ...
         num2str(M), '.mat'), 'model');
end

% Ensembles the AMLP
auxModel1 = advancedRandomInitialization(D, hiddenM);
for l = nlayers:-1:1
    auxModel1.W = auxModel1.W(1:l);
    auxModel1.B = auxModel1.B(1:l);
    auxModel2 = ensembleDeepNetwork(auxModel1, modelClassifier);
    [model, err] = mlpTrain(data, auxModel2, targets);
    save(strcat(currentPath, '/experiments/AMLP', num2str(l), '.', ...
         num2str(M), '.mat'), 'model');
end

% Ensembles the stacked AE
auxModel1 = greedyPretraining(data, hiddenM, F, outF, 'AE');
for l = nlayers:-1:1
    auxModel1.W = auxModel1.W(1:l);
    auxModel1.B = auxModel1.B(1:l);
    auxModel2 = ensembleDeepNetwork(auxModel1, modelClassifier);
    [model, err] = mlpTrain(data, auxModel2, targets);
    save(strcat(currentPath, '/experiments/AE', num2str(l), '.', ...
         num2str(M), '.mat'), 'model');
    activations = propagate(auxModel1, F, F, data);
    [auxModel3, err] = mlpTrain(cell2mat(activations(l)), modelClassifier, ...
                                targets);
    modelWFT = ensembleDeepNetwork(auxModel1, auxModel3);
    save(strcat(currentPath, '/experiments/AE', num2str(l), '.', ...
         num2str(M), 'WFT.mat'), 'modelWFT');
end

% Ensembles the stacked DAE
auxModel1 = greedyPretraining(data, hiddenM, F, outF, 'AE', 'noiseF', ...
                              @gaussian);
for l = nlayers:-1:1
    auxModel1.W = auxModel1.W(1:l);
    auxModel1.B = auxModel1.B(1:l);
    auxModel2 = ensembleDeepNetwork(auxModel1, modelClassifier);
    [model, err] = mlpTrain(data, auxModel2, targets);
    save(strcat(currentPath, '/experiments/DAE', num2str(l), '.', ...
         num2str(M), '.mat'), 'model');
    activations = propagate(auxModel1, F, F, data);
    [auxModel3, err] = mlpTrain(cell2mat(activations(l)), modelClassifier, ...
                                targets);
    modelWFT = ensembleDeepNetwork(auxModel1, auxModel3);
    save(strcat(currentPath, '/experiments/DAE', num2str(l), '.', ...
         num2str(M), 'WFT.mat'), 'modelWFT');
end

% Ensembles the stacked SAE
auxModel1 = greedyPretraining(data, hiddenM, F, outF, 'AE', 'sparsity', true);
for l = nlayers:-1:1
    auxModel1.W = auxModel1.W(1:l);
    auxModel1.B = auxModel1.B(1:l);
    auxModel2 = ensembleDeepNetwork(auxModel1, modelClassifier);
    [model, err] = mlpTrain(data, auxModel2, targets);
    save(strcat(currentPath, '/experiments/SAE', num2str(l), '.', ...
         num2str(M), '.mat'), 'model');
    activations = propagate(auxModel1, F, F, data);
    [auxModel3, err] = mlpTrain(cell2mat(activations(l)), modelClassifier, ...
                                targets);
    modelWFT = ensembleDeepNetwork(auxModel1, auxModel3);
    save(strcat(currentPath, '/experiments/SAE', num2str(l), '.', ...
         num2str(M), 'WFT.mat'), 'modelWFT');
end

% Ensembles the stacked SDAE
auxModel1 = greedyPretraining(data, hiddenM, F, outF, 'AE', 'sparsity', ...
                              true, 'noiseF', @gaussian);
for l = nlayers:-1:1
    auxModel1.W = auxModel1.W(1:l);
    auxModel1.B = auxModel1.B(1:l);
    auxModel2 = ensembleDeepNetwork(auxModel1, modelClassifier);
    [model, err] = mlpTrain(data, auxModel2, targets);
    save(strcat(currentPath, '/experiments/SDAE', num2str(l), '.', ...
         num2str(M), '.mat'), 'model');
    activations = propagate(auxModel1, F, F, data);
    [auxModel3, err] = mlpTrain(cell2mat(activations(l)), modelClassifier, ...
                                targets);
    modelWFT = ensembleDeepNetwork(auxModel1, auxModel3);
    save(strcat(currentPath, '/experiments/SDAE', num2str(l), '.', ...
         num2str(M), 'WFT.mat'), 'modelWFT');
end

% Ensembles the DBN
auxModel1 = greedyPretraining(data, hiddenM, F, F, 'RBM');
for l = nlayers:-1:1
    auxModel1.W = auxModel1.W(1:l);
    auxModel1.B = auxModel1.B(1:l);
    auxModel2 = ensembleDeepNetwork(auxModel1, modelClassifier);
    [model, err] = mlpTrain(data, auxModel2, targets);
    save(strcat(currentPath, '/experiments/RBM', num2str(l), '.', ...
         num2str(M), '.mat'), 'model');
    activations = propagate(auxModel1, F, F, data);
    [auxModel3, err] = mlpTrain(cell2mat(activations(l)), modelClassifier, ...
                                targets);
    modelWFT = ensembleDeepNetwork(auxModel1, auxModel3);
    save(strcat(currentPath, '/experiments/RBM', num2str(l), '.', ...
         num2str(M), 'WFT.mat'), 'modelWFT');
end
