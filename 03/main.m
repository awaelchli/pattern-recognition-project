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

% load test mat file with cut words
fprintf('Loading test images.\n');
load('preprocessing/cut_wordsTest.mat');

% binarize images
fprintf('Binarizing test images.\n');
[binarizedImagesTest] = binarize_images(cutWords);

%%
% imshow(uint8(cutWords{1, 2}{2}));
% figure();
% imshow(binarizedImages{1, 2}{2});

%% Compute features with sliding window
% Simple example with one image
normsize = [100, 100];
test = [1:5];
r_constraint = 15;

fileTranscriptionID = fopen('data/ground-truth/transcription.txt','r');
transcription = textscan(fileTranscriptionID,'%d-%s %s');

fileID = fopen('results.txt','w');
    
kwID = fopen('TestKWS/task/keywords.txt','r');
kw = textscan(kwID,'%s');
for index = 1:size(kw{1},1)
    index
    keyword = char(kw{1}(index));

    keywordParts = strsplit(keyword,',');
    kwName = char(keywordParts(1));
    kwID = char(keywordParts(2));
    kwPositions = strsplit(char(keywordParts(2)),'-');
    fileName = char(strcat(kwPositions(1),'.svg'));
    fprintf(fileID,'%s',kwName);

%     kwIndexesArray = strfind(transcription{1,3},keyword);
%     kwIndexes = find(not(cellfun('isempty', kwIndexesArray)));
%     transIndex = kwIndexes(1);
%     fileName = strcat(num2str(transcription{1,1}(transIndex)),'.svg');
    
    biIndexesArray = strfind(binarizedImages(:,1),fileName);
    biIndexes = find(not(cellfun('isempty', biIndexesArray)));
    biIndex = biIndexes(1);
    fileImages = binarizedImages{biIndex,2};
    
    fileIDs = binarizedImages{biIndex,3};
    imageIDArray = strfind(cellstr(fileIDs),kwID);
    imageIndexes = find(not(cellfun('isempty', imageIDArray)));
    imageIndex = imageIndexes(1);
    kwImage=fileImages{imageIndex};
%     transcriptionAtName = find(transcription{1}==transcription{1,1}(transIndex));
%     transStartIndex = transcriptionAtName(1);
%     imageNumber = transIndex-(transStartIndex-1);
%     kwImage=fileImages{imageNumber};
    
    %alexandria = binarizedImages{4,2}{129};
    kwImage = fake_imresize(kwImage, normsize);
    featuresKw = sliding_window(kwImage, 1, 0);

    %files from test.txt 
    for i=test

%         nameString = strsplit(binarizedImagesTest{i,1},'.');
%         name = char(nameString(1));
%         transcriptionAtName = find(transcription{1}==str2num(name));
%         transStartIndex = transcriptionAtName(1);
        
        for j=1:size(binarizedImagesTest{i,2},1)
            %testImageID = strcat(num2str(transcription{1,1}(transStartIndex+j-1)),'-',char(transcription{1,2}(transStartIndex+j-1)));
            testImageID = binarizedImagesTest{i, 3}(j,:);

            
            testImage = binarizedImagesTest{i, 2}{j};
            testImage = fake_imresize(testImage, normsize);
            featuresTest = sliding_window(testImage, 1, 0);

            [ path, pathCost, matrix ] = dynamic_time_warp(featuresKw, featuresTest, r_constraint);

            fprintf(fileID,', %s, %0.2f',testImageID,pathCost);
        end
    end
    fprintf(fileID,'\n');

end
fclose(fileID);


%% Some figures for testing
% image = binarizedImages{1, 2}{2};
% 
% window_size = 1;
% window_offset = 0;
% 
% features = sliding_window(image, window_size, window_offset);
% 
% figure;
% imshow(image);
% title('binarized image');
% 
% figure;
% plot(features(1, :));
% title('upper contour feature');
% 
% figure;
% plot(features(2, :));
% title('lower contour feature');

%% Dynamic Time Warping
% Reference: http://ciir-publications.cs.umass.edu/pdf/MM-38.pdf
%
% Simple example: warping two images (using lower contour)

% image1 = binarizedImages{1, 2}{2};
% image2 = binarizedImages{3, 2}{2};
% 
% % Normalize image sizes
% normsize = [100, 100];
% image1 = fake_imresize(image1, normsize);
% image2 = fake_imresize(image2, normsize);
% 
% features1 = sliding_window(image1, 1, 0);
% features2 = sliding_window(image2, 1, 0);
% 
% % "Radius" of diagonal constraint
% r_constraint = 15;
% 
% [ path, pathCost, matrix ] = dynamic_time_warp(features1, features2, r_constraint);
% pathCost
% figure;
% imshow(matrix, []);
% title('Matrix');
% 
% figure;
% subplot(2, 2, 1);
% imshow(image1);
% subplot(2, 2, 2);
% imshow(image2);
% subplot(2, 2, 3);
% plot(features1(1, :));
% subplot(2, 2, 4);
% plot(features2(1, :));
% 
% figure;
% plot(path(2, :), path(1, :));
% title('path after dynamic time warping');