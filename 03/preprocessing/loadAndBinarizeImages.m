function [ images, binarizedImages ] = loadAndBinarizeImages()
%   Detailed explanation goes here
imagesFolder = 'data/images/';
imageFiles = dir([imagesFolder '*.jpg']);
nFiles = length(imageFiles);
images = cell(nFiles, 1);
binarizedImages = cell(nFiles, 1);

for k=1:nFiles
    images{k} = imread([imagesFolder imageFiles(k).name]);
    % binarize images
    binarizedImages{k} = imbinarize(images{k}, 0.75);
end

end

