function [ feature ] = upper_contour( window_content )
%UPPER_CONTOUR Computes the feature of the upper contour in a window
%
%   Input:  	The window content, a slice of the image. It is assumed
%               that the contents are binary with background set to 0 and
%               foreground set to 1.
%
%   Output:     The feature is the height profile along the upper contour
%               normalized to the range [0, 1].

idx = find(sum(window_content, 2) ~= 0, 1, 'first');

if(isempty(idx))
    % No contour present
    feature = 0;
else
    feature = size(window_content, 1) - idx;
end

% Normalize
feature = feature / size(window_content, 1);

end

