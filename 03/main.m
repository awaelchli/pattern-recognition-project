%% Pattern Recognition project 3

close all;

% load and binarize images
if ~exist('images', 'var')
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

imshow(binarizedImages{2});