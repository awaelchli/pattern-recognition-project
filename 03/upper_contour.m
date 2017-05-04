function [ feature ] = upper_contour( window_content )
%UPPER_CONTOUR Computes the feature of the upper contour in a window
%
%   Input:  	The window content, a slice of the image. It is assumed
%               that the contents are binary with background set to 0 and
%               foreground set to 1.
%
%   Output:     The feature is a vector of size h x 1, where h is the
%               height of the image. The vector contains a 1 where the
%               upper contour is present, and zeros everywhere else.

feature = zeros(size(window_content, 1), 1);

if (sum(sum(window_content)) > 0)
    % Upper contour is present
    
    [ ~, idx ] = max(sum(window_content, 2) > 0);
    feature(idx) = 1;
end

end

