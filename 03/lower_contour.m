function [ feature ] = lower_contour( window_content )
%LOWER_CONTOUR Computes the feature of the lower contour in a window
%
%   Input:  	The window content, a slice of the image. It is assumed
%               that the contents are binary with background set to 0 and
%               foreground set to 1.
%
%   Output:     The feature is the height profile along the lower contour
%               normalized to the range [0, 1].

% Compute upper contour on vertically flipped image
feature = upper_contour(window_content(end : -1 : 1, :));

end

