function [ feature ] = projected_profile( window_content )
%PROJECTED_PROFILE Computes the projected profile of a window content
%
%   Input:  	The window content, a slice of the image. It is assumed
%               that the contents are binary with background set to 0 and
%               foreground set to 1.
%
%   Output:     The feature is the projected profile normalized to the
%               range [0, 1].

feature = sum(sum(window_content)) / numel(window_content);

end

