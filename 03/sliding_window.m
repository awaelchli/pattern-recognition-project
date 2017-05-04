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
%   Outputs:    features:       H x T x 2 matrix, where H is the size of
%                               the image, T is the number of windows that 
%                               were produced. The last dimension separates
%                               different feature types, currently
%                               supported are upper- and lower contour.


% invert image
image = 1 - image;

width = size(image, 2);
height = size(image, 1);

% Compute all locations of the window (along horizontal axis)
locations = 1 : window_offset + window_size : width;

feature_types = 2;
features = zeros(height, size(locations, 2), feature_types);

for i = 1 : size(locations, 2)
    
    % Extract window content at current location
    l = locations(i);
    window_content = image(:, l : (l + window_size - 1));
    
    features(:, i, 1) = upper_contour(window_content);
    features(:, i, 2) = lower_contour(window_content);
end


end

