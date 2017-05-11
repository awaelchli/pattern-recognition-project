%% Pattern Recognition Project 3

close all;

addpath('preprocessing');
addpath('feature_extraction');

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
plot(features(1, :));
title('upper contour feature');

figure;
plot(features(2, :));
title('lower contour feature');

%% Dynamic Time Warping
% Reference: http://ciir-publications.cs.umass.edu/pdf/MM-38.pdf
%
% Simple example: warping two images (using lower contour)

image1 = binarizedImages{1, 2}{2};
image2 = binarizedImages{3, 2}{2};

% Normalize image sizes
normsize = [100, 100];
image1 = fake_imresize(image1, normsize);
image2 = fake_imresize(image2, normsize);

features1 = sliding_window(image1, 1, 0);
features2 = sliding_window(image2, 1, 0);

% "Radius" of diagonal constraint
r_constraint = 15;

[ path, pathCost, matrix ] = dynamic_time_warp(features1, features2, r_constraint);

figure;
imshow(matrix, []);

figure;
subplot(2, 2, 1);
imshow(image1);
subplot(2, 2, 2);
imshow(image2);
subplot(2, 2, 3);
plot(features1(1, :));
subplot(2, 2, 4);
plot(features2(1, :));

figure;
plot(path(2, :), path(1, :));
title('path after dynamic time warping');