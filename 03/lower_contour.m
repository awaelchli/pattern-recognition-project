function [ feature ] = lower_contour( window_content )
%LOWER_CONTOUR Computes the feature of the lower contour in a window
%
%   Input:  	The window content, a slice of the image. It is assumed
%               that the contents are binary with background set to 0 and
%               foreground set to 1.
%
%   Output:     The feature is a vector of size h x 1, where h is the
%               height of the image. The vector contains a 1 where the
%               lower contour is present, and zeros everywhere else.

% Compute upper contour on vertically flipped image
feature = upper_contour(window_content(end : -1: 1, :));
feature = feature(end : -1 : 1);

end

