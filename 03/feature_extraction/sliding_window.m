function [ features ] = sliding_window( image, window_size, window_offset )
%SLIDING_WINDOW Computes features with a sliding window
%
%   Inputs:     image:          Binarized 2D image. Values for background
%                               are expected to be 1 (white) and forground 
%                               values 0 (black)
%
%               window_size:    The size of the window in horizontal
%                               direction. The size in vertical direction
%                               will be set to the image height.
%
%               window_offset:  Offset when moving the window, by default 
%                               it is set to 0.
%           
%   Outputs:    features:       TODO: doc


% Invert image
image = 1 - image;

width = size(image, 2);
height = size(image, 1);

% Compute all locations of the window (along horizontal axis)
locations = 1 : window_offset + window_size : width;

feature_types = 2;
features = zeros(feature_types, size(locations, 2));

for i = 1 : size(locations, 2)
    
    % Extract window content at current location
    l = locations(i);
    window_content = image(:, l : (l + window_size - 1));
    
    % Concatenate features
    features(1, i) = upper_contour(window_content);
    features(2, i) = lower_contour(window_content);
    features(3, i) = projected_profile(window_content);
end


end

