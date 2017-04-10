%%
clear all;
close all;

%% settings
cList = [1];         % learning rate
nbrOfHiddenNodesList = [100];
nbrOfOutputNodes = 10;
nbrOfEpoches = 100;
lambdaList = [0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3];                 % regularization
fraction = 1;

%% Load the training set

train = csvread('../../data/train.csv');
n_data = ceil(fraction * size(train, 1));
trainLabels = train(1 : n_data, 1);
trainData = train(1 : n_data, 2 : end);
clear train;

%% Randomly select 100 data points to display
sel = randperm(size(trainData, 1));
sel = sel(1:100);

displayData(trainData(sel, :));

%% Feature Selection/ Feature Descriptor
fprintf('\nTransform trainingset to hog\n');
for i=1:size(trainData,1)
    hg=hog(reshape(trainData(i,:),28,28),8);
    newData(i,:)= hg(:);
end
trainData=newData;

%% Load the test set
test = csvread('../../data/test.csv');
n_data = ceil(fraction * size(test, 1));
testLabels = test(1 : n_data, 1);
testData = test(1 : n_data, 2 : end);
clear test;

%% Feature Selection/ Feature Descriptor
fprintf('\nTransform testset to hog\n');
clear newData;
for i=1:size(testData,1)
    hg=hog(reshape(testData(i,:),28,28),8);
    newData(i,:)= hg(:);
end
testData=newData;

%% train and test
%save('results/config.mat', 'cList', 'nbrOfHiddenNodesList', 'lambdaList');
%i = 0;
%for nbrOfHiddenNodes = nbrOfHiddenNodesList
%    for lambda = lambdaList
%        for c = cList
%            i=i+1;
            % train 
%            tic;
%            [Theta1, Theta2, J] = mlpTrain(trainLabels, trainData, c, nbrOfHiddenNodes, nbrOfOutputNodes, nbrOfEpoches, lambda);
%            toc;
%
%             save(['results/lambda',num2str(lambda,2),'c',num2str(c,2),'hn', num2str(nbrOfHiddenNodes,1),'i', num2str(i,1),'model.mat'], 'Theta1', 'Theta2');
%             
%             figure;
%             plot(J);
%             hold on;
%             title(['lambda=',num2str(lambda,2),' c=',num2str(c,2),' hn=', num2str(nbrOfHiddenNodes,1)]);
%             xlabel('Epoche');
%             ylabel('Error');
%             saveas(gcf,['plots/lambda',num2str(lambda,2),'c',num2str(c,2),'hn', num2str(nbrOfHiddenNodes,1),'i', num2str(i,1),'error.','png']);
%             hold off;
%             close gcf; 
% 
%             % classify
%             % training set accuracy
%             pred = mlpPredict(Theta1, Theta2, trainData);
%             trainingsetAccuracy = mean(double(pred == trainLabels)) * 100;
%             fprintf('\nhn: %f c: %f lambda: %f Training Set Accuracy: %f\n', nbrOfHiddenNodes, c,lambda, trainingsetAccuracy);
% 
%             % test set accuracy
%             pred = mlpPredict(Theta1, Theta2, testData);
%             testestAccuracy = mean(double(pred == testLabels)) * 100;
%             fprintf('\nhn: %f c: %f lambda: %f Test Set Accuracy: %f\n',nbrOfHiddenNodes, c,lambda, testestAccuracy);
%             save(['results/lambda',num2str(lambda,2),'c',num2str(c,2),'hn', num2str(nbrOfHiddenNodes,1),'i',num2str(i,1),'accuracies.mat'], 'testestAccuracy', 'trainingsetAccuracy');
% 
%         end
%     end
% end

%% split trainingset in crossvalidation
fraction = 0.8;
n_cross = ceil(fraction * size(trainData, 1));
crossValData = trainData(1 : n_cross, :);
crossValLabels = trainLabels(1 : n_cross);
trainData = trainData(n_cross+1 : end, :);
trainLabels = trainLabels(n_cross+1: end);

%% cross val
for nbrOfHiddenNodes = nbrOfHiddenNodesList
    for c = cList
    	[error_train, error_xval, error_test, accuracy_train, accuracy_xval, accuracy_test] = validationCurve(trainData, trainLabels, crossValData, crossValLabels, testData, testLabels, nbrOfOutputNodes, nbrOfHiddenNodes, lambdaList, c, nbrOfEpoches);
    
        % plot 
        figure;
        title(sprintf('Regularization Crossvalidation\nc=%.2f hn=%.0f i=%.0f',c,nbrOfHiddenNodes,i));
        yyaxis left;
        plot(lambdaList, error_train, lambdaList, error_xval, lambdaList, error_test, 'LineWidth', 2);
        xlabel('regularization (lambda)');
        ylabel('cost');
        legend('training set', 'validation set', 'test set');
        hold on;
        yyaxis right;
        ylabel('accuracy');
        plot(lambdaList, accuracy_train, lambdaList, accuracy_xval, lambdaList, accuracy_test, 'LineWidth', 2);
        legend('training set', 'validation set', 'test set');
        hold off;
        saveas(gcf,['plots/xval_c',num2str(c,2),'_hn', num2str(nbrOfHiddenNodes,1),'i', num2str(i,1),'.','png']);
        close gcf; 

    end
end



  