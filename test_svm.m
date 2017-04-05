%% Pattern Recognition FS2017
%  SVM Test
clc;
clear;

%% Load the training set
fraction = 0.1;
train = csvread('data/train.csv');
n_data = ceil(fraction * size(train, 1));
train_labels = train(1 : n_data, 1);
train_images = train(1 : n_data, 2 : end);
clear train;

%% Feature Selection/ Feature Descriptor
for i=1:size(train_images,1)
    hg=hog(reshape(train_images(i,:),28,28),8);
    newData(i,:)= hg(:);
end
size(newData)
train_images=newData;

%%  Train model
tic;
model = svmtrain(train_labels, train_images);
toc