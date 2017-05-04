%% Pattern Recognition Project 3

close all;

addpath('preprocessing');

% load mat file with cut words
fprintf('Loading images.\n');
load('preprocessing/cut_words.mat');

% binarize images
fprintf('Binarizing images.\n');
[binarizedImages] = binarize_images(cutWords);

%%
imshow(uint8(cutWords{1, 2}{2}));
figure();
imshow(binarizedImages{1, 2}{2});


%% Compute features with sliding window
% Simple example with one image

image = binarizedImages{1, 2}{2};

window_size = 1;
window_offset = 0;

features = sliding_window(image, window_size, window_offset);

figure;
imshow(image);
title('binarized image');

figure;
imshow(squeeze(features(:, :, 1)));
title('upper contour feature');

figure;
imshow(squeeze(features(:, :, 2)));
title('lower contour feature');