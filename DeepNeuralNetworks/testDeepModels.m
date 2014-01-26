% Deep neural networks test.
%
% Author:
%	David Diaz Vico

% Loads the PCP
load(strcat('PCP1.',num2str(M),'.mat'));

% Uses the PCP as a classifier
activations = propagate(model,F,outF,testdata);
predictedtargets = cell2mat(activations(2));
predictedlabels = targets2labels(predictedtargets);
errors = predictedlabels~=testlabels;
error = sum(errors)/length(errors);
save(strcat(currentPath,'/experiments/PCP1.',num2str(M),...
	'test.mat'),'testdata','testlabels','predictedtargets',...
	'predictedlabels','error');
error

% Visualizes the propagation
visualizePropagation(1,model.W,testdata,activations,...
	strcat(strcat('PropagationPCP1.',num2str(M))),...
	strcat(currentPath,'/experiments'))

for l = 1:nlayers

	L = Larray(l)+2;

	% Loads the MLP
	load(strcat('MLP',num2str(l),'.',num2str(M),'.mat'));

	% Uses the MLP as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/MLP',num2str(l),'.',num2str(M),...
		'test.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorMLP(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+1,model.W,testdata,activations,...
		strcat(strcat('PropagationMLP',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the AMLP
	load(strcat('AMLP',num2str(l),'.',num2str(M),'.mat'));

	% Uses the AMLP as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/AMLP',num2str(l),'.',num2str(M),...
		'test.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorAMLP(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+2,model.W,testdata,activations,...
		strcat(strcat('PropagationAMLP',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the AE
	load(strcat('AE',num2str(l),'.',num2str(M),'.mat'));

	% Uses the AMLP as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/AE',num2str(l),'.',num2str(M),...
		'test.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorAE(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+3,model.W,testdata,activations,...
		strcat(strcat('PropagationAE',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the AE without fine-tuning
	load(strcat('AE',num2str(l),'.',num2str(M),'WFT.mat'));

	% Uses the AE as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/AE',num2str(l),'.',num2str(M),...
		'WFTtest.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorAEWFT(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+4,model.W,testdata,activations,...
		strcat(strcat('PropagationAEWFT',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the DAE
	load(strcat('DAE',num2str(l),'.',num2str(M),'.mat'));

	% Uses the DAE as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/DAE',num2str(l),'.',num2str(M),...
		'test.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorDAE(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+5,model.W,testdata,activations,...
		strcat(strcat('PropagationDAE',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the DAE without fine-tuning
	load(strcat('DAE',num2str(l),'.',num2str(M),'WFT.mat'));

	% Uses the DAE as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/DAE',num2str(l),'.',num2str(M),...
		'WFTtest.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorDAEWFT(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+6,model.W,testdata,activations,...
		strcat(strcat('PropagationDAEWFT',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the SAE
	load(strcat('SAE',num2str(l),'.',num2str(M),'.mat'));

	% Uses the SAE as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/SAE',num2str(l),'.',num2str(M),...
		'test.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorSAE(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+7,model.W,testdata,activations,...
		strcat(strcat('PropagationSAE',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the SAE without fine-tuning
	load(strcat('SAE',num2str(l),'.',num2str(M),'WFT.mat'));

	% Uses the SAE as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/SAE',num2str(l),'.',num2str(M),...
		'WFTtest.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorSAEWFT(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+8,model.W,testdata,activations,...
		strcat(strcat('PropagationSAEWFT',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the SDAE
	load(strcat('SDAE',num2str(l),'.',num2str(M),'.mat'));

	% Uses the SDAE as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/SDAE',num2str(l),'.',num2str(M),...
		'test.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorSDAE(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+9,model.W,testdata,activations,...
		strcat(strcat('PropagationSDAE',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the SDAE without fine-tuning
	load(strcat('SDAE',num2str(l),'.',num2str(M),'WFT.mat'));

	% Uses the SDAE as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/SDAE',num2str(l),'.',num2str(M),...
		'WFTtest.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorSDAEWFT(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+10,model.W,testdata,activations,...
		strcat(strcat('PropagationSDAEWFT',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the RBM
	load(strcat('RBM',num2str(l),'.',num2str(M),'.mat'));

	% Uses the RBM as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/RBM',num2str(l),'.',num2str(M),...
		'test.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorRBM(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+11,model.W,testdata,activations,...
		strcat(strcat('PropagationDBN',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

	% Loads the RBM without fine-tuning
	load(strcat('RBM',num2str(l),'.',num2str(M),'WFT.mat'));

	% Uses the RBM as a classifier
	activations = propagate(model,F,outF,testdata);
	predictedtargets = cell2mat(activations(L));
	predictedlabels = targets2labels(predictedtargets);
	errors = predictedlabels~=testlabels;
	error = sum(errors)/length(errors);
	save(strcat(currentPath,'/experiments/RBM',num2str(l),'.',num2str(M),...
		'WFTtest.mat'),'testdata','testlabels','predictedtargets',...
		'predictedlabels','error');
	errorRBMWFT(l) = error;

	% Visualizes the propagation
	visualizePropagation(12*(l-1)+12,model.W,testdata,activations,...
		strcat(strcat('PropagationDBNWFT',num2str(l),'.',num2str(M))),...
		strcat(currentPath,'/experiments'))

end

% Plots the classification errors
figure(12*(l-1)+13,'Name','Classification error');
%plot(Larray,errorMLP,Larray,errorAMLP,Larray,errorAE,'-',Larray,errorDAE,'-',Larray,errorSAE,'-',Larray,errorSDAE,'-',Larray,errorRBM,'-',Larray,errorAEWFT,'--',Larray,errorDAEWFT,'--',Larray,errorSAEWFT,'--',Larray,errorSDAEWFT,'--',Larray,errorRBMWFT,'--');
plot(Larray,errorMLP,'k^-',Larray,errorAMLP,'ko-',Larray,errorAE,'r^-',Larray,errorDAE,'g^-',Larray,errorSAE,'b^-',Larray,errorSDAE,'y^-',Larray,errorRBM,'m^-',Larray,errorAEWFT,'ro-',Larray,errorDAEWFT,'go-',Larray,errorSAEWFT,'bo-',Larray,errorSDAEWFT,'yo-',Larray,errorRBMWFT,'mo-');
legend('MLP','AMLP','AE','DAE','SAE','SDAE','RBM','AEWFT','DAEWFT','SAEWFT','SDAEWFT','RBMWFT');
xlabel('Number of hidden layers');
ylabel('Classification error');
title('Classification error');
print('-djpeg',strcat(currentPath,strcat(...
	'/experiments/ClassificationError',num2str(M),'.jpg')));

