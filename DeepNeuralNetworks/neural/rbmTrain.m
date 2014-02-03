function [model, errors] = rbmTrain(X, M, varargin)
% Trains an RBM.
%
% Arguments:
%	X		the training data
%	M		the number of hidden units
%	varargin	aditional inputs
%			(specified as name value pairs or in struct)
%		method		CD or SML
%		eta		learning rate
%		momentum	momentum for smoothness amd to prevent
%				overfitting
%				momentum is not recommended with SML
%		maxepoch	# of epochs: each is a full pass through train
%				data
%		avglast		how many epochs before maxepoch to start
%				averaging before. Procedure suggested for faster
%				convergence by Kevin Swersky in his MSc thesis
%		penalty		weight decay factor
%		weightdecay	a boolean flag. When set to true, the weights
%				are decayed linearly from penalty->0.1*penalty
%				in epochs
%		batchsize	the number of training instances per batch
%		verbose		for printing progress
%		anneal		a boolean flag. If set true, the penalty is
%				annealed linearly through epochs to 10% of its
%				original value
%		F		the activation function in the hidden layers
%		outF		the activation function in the output layer
%
% Returns:
%	model	the RBM model consisting of W and B
%	errors	the errors in reconstruction at every epoch
%
% Author:
%	David Diaz Vico
%	Based on the implementation of RBMLIB by Andrej Karpathy
%	http://code.google.com/p/matrbm/

% Process options
args = prepareArgs(varargin);
[method...
 eta...
 momentum...
 maxepoch...
 avglast...
 penalty...
 batchsize...
 verbose...
 anneal...
 F...
 outF...
] = process_options(args, ...
                    'method', 'CD', ...
                    'eta', 0.1, ...
                    'momentum', 0.5, ...
                    'maxepoch', 50, ...
                    'avglast', 5, ...
                    'penalty', 2e-4, ...
                    'batchsize', 100, ...
                    'verbose', false, ...
                    'anneal', false, ...
                    'F', @logistic, ...
                    'outF', @logistic);

avgstart = maxepoch - avglast;
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
end

% Fit RBM
W = 0.1*randn(D, M);
c = zeros(D, 1);
b = zeros(M, 1);
ph = zeros(M, N);
nh = zeros(M, N);
phstates = zeros(M, N);
nhstates = zeros(M, N);
negdata = zeros(D, N);
negdatastates = zeros(D, N);
Winc = zeros(D, M);
binc = zeros(M, 1);
cinc = zeros(D, 1);
Wavg = W;
bavg = b;
cavg = c;
t = 1;
errors = zeros(1, maxepoch);

for epoch = 1:maxepoch
    errsum = 0;
    if (anneal)
        penalty = oldpenalty - 0.9*epoch/maxepoch*oldpenalty;
    end
    for batch = 1:numbatches
        rata = batchdata{batch};
        [D, N] = size(data);
        % Go up
        ph = F(W'*data + repmat(b, 1, N));
        phstates = ph > rand(M, N);
        if (isequal(method, 'SML'))
 			if (epoch == 1 && batch == 1)
 				nhstates = phstates;
            end
        elseif (isequal(method,'CD'))
            nhstates = phstates;
        end
        % Go down
        negdata = outF(W*nhstates + repmat(c, 1, N));
        negdatastates = negdata > rand(D, N);
        % Go up one more time
        nh = F(W'*negdatastates + repmat(b, 1, N));
        nhstates = nh > rand(M, N);
        % Update weights and biases
        dW = (data*ph' - negdatastates*nh');
        dc = sum(data, 2) - sum(negdatastates, 2);
        db = sum(ph, 2) - sum(nh, 2);
        Winc = momentum*Winc + eta*(dW/N - penalty*W);
        binc = momentum*binc + eta*(db/N);
        cinc = momentum*cinc + eta*(dc/N);
        W = W + Winc;
        b = b + binc;
        c = c + cinc;
        if (epoch > avgstart)
            % Apply averaging
            Wavg = Wavg - (1/t)*(Wavg - W);
            cavg = cavg - (1/t)*(cavg - c);
            bavg = bavg - (1/t)*(bavg - b);
            t = t + 1;
        else
            Wavg = W;
            bavg = b;
            cavg = c;
        end
        % Accumulate reconstruction error
        err = sum(sum((data - negdata).^2/2));
        errsum = err + errsum;
    end
    errors(epoch) = errsum;
    if (verbose)
        fprintf('Ended epoch %i/%i, reconsruction error is %f\n', epoch, ...
                maxepoch, errsum);
    end
end

Wcell = cell(2);
Bcell = cell(2);
Wcell(1) = Wavg;
Wcell(2) = Wavg';
Bcell(1) = bavg;
Bcell(2) = zeros(D, 1);
model.W = Wcell;
model.B = Bcell;
