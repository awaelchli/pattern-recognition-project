function [ binarizedImages ] = binarize_images(cutWords)
nFiles = size(cutWords, 1);
binarizedImages = cutWords;

% binarize images
for k=1:nFiles
    nWords = length(cutWords{k, 2});
    binarizedImages{k, 2} = cell(nWords, 1);
    for l=1:nWords
        threshold = 0.70;
        absoluteThreshold = threshold * 255;
        binarizedImages{k, 2}{l} = cutWords{k, 2}{l} > absoluteThreshold;
    end
   
end

end

