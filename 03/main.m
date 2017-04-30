%% Pattern Recognition project 3

close all;

addpath('preprocessing');

% load mat file with cut words
fprintf('Loading images.');
load('preprocessing/cut_words.mat');

% binarize images
fprintf('Binarizing images.');
[binarizedImages] = binarize_images(cutWords);

imshow(binarizedImages{1}{2});