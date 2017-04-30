function [ binarizedImages ] = binarize_images(cutWords)
nFiles = length(cutWords);
binarizedImages = cell(nFiles, 1);

% binarize images
for k=1:nFiles
    nWords = length(cutWords{k});
    binarizedImages{k} = cell(nWords, 1);
    for l=1:nWords
        threshold = 0.70;
        absoluteThreshold = threshold * 255;
        binarizedImages{k}{l} = cutWords{k}{l} > absoluteThreshold;
    end
   
end

end

