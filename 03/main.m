%% Pattern Recognition project 3

close all;

addpath('preprocessing');

% load and binarize images
fprintf('Loading and binarizing images.');
[images, binarizedImages] = loadAndBinarizeImages();

imshow(binarizedImages{2});