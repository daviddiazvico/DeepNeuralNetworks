function [model, errors] = mlpTrain(X, model, targets, varargin)
% Trains a MLP.
%
% Arguments:
%	X		the training data
%	model		the initial model of the MLP, with initial W and B
%	targets		the training targets
%	varargin	aditional inputs
%			(specified as name value pairs or in struct)
%		eta		learning rate
%		maxepoch	# of epochs: each is a full pass through train
%				data
%		penalty		weight decay factor
%		weightdecay	a boolean flag. When set to true, the weights
%				are decayed linearly from penalty->0.1*penalty
%				in epochs
%		sparsity	a boolean flag. When set to true, sparse
%				activations are forced
%		spfactor	sparsity factor
%		spactiv		sparsity mean activation
%		noiseF		noise function
%		noisefactor	noise factor: gaussian squared sigma, masking or
%				saltpepper change factor
%		batchsize	the number of training instances per batch
%		verbose		for printing progress
%		F		the activation function in the hidden layers
%		dF		the derivative of the activation funcion in the
%				hidden layers
%		outF		the activation function in the output layer
%		doutF		the derivative of the activation function in the
%				output layer
%
% Returns:
%	model	the MLP model consisting of W and B
%	errors	the errors at every epoch
%
% Author:
%	David Diaz Vico

% Process options
args = prepareArgs(varargin);
[eta ...
 maxepoch ...
 penalty ...
 weightdecay ...
 sparsity ...
 spfactor ...
 spactiv ...
 noiseF ...
 noisefactor ...
 batchsize ...
 verbose ...
 F ...
 dF ...
 outF ...
 doutF ...
] = process_options(args, ...
                    'eta', 0.1, ...
                    'maxepoch', 50, ...
                    'penalty', 2e-4, ...
                    'weightdecay', true, ...
                    'sparsity', false, ...
                    'spfactor', 0.2, ...
                    'spactiv', 0.2, ...
                    'noiseF', @none, ...
                    'noisefactor', 0.25, ...
                    'batchsize', 50, ...
                    'verbose', false, ...
                    'F', @logistic, ...
                    'dF', @dlogistic, ...
                    'outF', @identity, ...
                    'doutF', @didentity);

oldpenalty = penalty;
[D, N] = size(X);

if (verbose)
    fprintf('Preprocessing data...\n')
end

% Create batches
numbatches = ceil(N/batchsize);
groups = repmat(1:numbatches, 1, batchsize);
groups = groups(1:N);
groups = groups(randperm(N));
for i = 1:numbatches
    batchdata{i} = X(:, groups == i);
    batchtargets{i} = targets(:, groups == i);
end

% Fit MLP
L = length(model.W);
W = model.W;
B = model.B;
dW = cell(1, L);
dB = cell(1, L);
dWsp = cell(1, L);
dBsp = cell(1, L);
Y = cell(1, L);
error = cell(1,L);
t = 1;
errors = zeros(1, maxepoch);

for epoch = 1:maxepoch
    errsum = 0;
    for batch = 1:numbatches
        data = batchdata{batch};
        [D, N] = size(data);
        % Corrupt data
        corrupteddata = noiseF(data, noisefactor);
        % Propagate
        model.W = W;
        model.B = B;
        Y = propagate(model, F, outF, corrupteddata);
        % Backpropagate through the last layer
        prevY = cell2mat(Y(L - 1));
        thisY = cell2mat(Y(L));
        error(L) = batchtargets{batch} - thisY;
        thiserror = cell2mat(error(L));
        dW(L) = (thiserror*prevY')';
        dB(L) = sum(thiserror, 2);
        dWsp(L) = 0;
        dBsp(L) = 0;
        % Backpropagate through the hidden layers
        for l = L - 1:-1:2
            prevY = cell2mat(Y(l - 1));
            thisW = cell2mat(W(l));
            nextW = cell2mat(W(l + 1));
            thisB = repmat(cell2mat(B(l)), 1, N);
            nexterror = cell2mat(error(l + 1));
            error(l) = nextW*nexterror;
            thiserror = cell2mat(error(l));
            dY = dF(thisW'*prevY + thisB);
            dW(l) = prevY*(thiserror.*dY)';
            dB(l) = sum(thiserror.*dY, 2);
            if (sparsity == true)
                prevY = cell2mat(Y(l - 1));
                thisY = cell2mat(Y(l));
                thisW = cell2mat(W(l));
                thisB = repmat(cell2mat(B(l)), 1, N);
                mY = sum(thisY, 2)/N;
                spcoeff = -spactiv./mY + (1 - spactiv)./(1 - mY);
                dWsp(l) = spfactor*(prevY*(dF(thisW'*prevY + thisB))').* ...
                                    repmat(spcoeff', D, 1);
                dBsp(l) = spfactor*(dF(thisW'*prevY + thisB)*ones(N, 1)).* ...
                                    spcoeff;
            else
                dWsp(l) = 0;
                dBsp(l) = 0;
            end
        end
        % Backpropagate through the first layer
        thisW = cell2mat(W(1));
        nextW = cell2mat(W(2));
        thisB = repmat(cell2mat(B(1)), 1, N);
        nexterror = cell2mat(error(2));
        error(1) = nextW*nexterror;
        thiserror = cell2mat(error(1));
        dY = dF(thisW'*data + thisB);
        dW(1) = data*(thiserror.*dY)';
        dB(1) = sum(thiserror.*dY,2);
        if (sparsity == true)
            thisY = cell2mat(Y(1));
            thisW = cell2mat(W(1));
            thisB = repmat(cell2mat(B(1)), 1, N);
            mY = sum(thisY, 2)/N;
            spcoeff = (-spactiv./mY + (1 - spactiv)./(1 - mY));
            dWsp(1) = spfactor*(data*(dF(thisW'*data + thisB))').*repmat( ...
                                spcoeff', D, 1);
            dBsp(1) = spfactor*(dF(thisW'*data + thisB)*ones(N, 1)).*spcoeff;
        else
            dWsp(1) = 0;
            dBsp(1) = 0;
        end
        % Update weights and biases
        for l = 1:L
            W(l) = cell2mat(W(l)) + eta*(cell2mat(dW(l))/N - penalty* ...
                   cell2mat(W(l)) - cell2mat(dWsp(l)));
            B(l) = cell2mat(B(l)) + eta*(cell2mat(dB(l))/N - cell2mat(dBsp(l)));
        end
        % Accumulate reconstruction error
        lastY = cell2mat(Y(L));
        err = sum(sum((batchtargets{batch} - lastY).^2/2));
        errsum = err + errsum;
    end
    errors(epoch) = errsum;
    if (verbose)
        fprintf('Ended epoch %i/%i, Reconsruction error is %f\n', epoch, ...
                maxepoch, errsum);
    end
end

model.W = W;
model.B = B;
