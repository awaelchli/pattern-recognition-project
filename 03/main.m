%% Pattern Recognition project 3

close all;

addpath('preprocessing');

% load mat file with cut words
fprintf('Loading images.');
load('preprocessing/cut_words.mat');

% binarize images
fprintf('Binarizing images.');
[binarizedImages] = binarize_images(cutWords);

imshow(uint8(cutWords{1, 2}{2}));
figure();
imshow(binarizedImages{1, 2}{2});