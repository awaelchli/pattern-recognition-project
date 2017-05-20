function [ band_mask ] = sakoe_chiba_band( height, width, r )
%SAKOE_CHIBA_BAND Create a Sakoe-Chiba mask for an m by n matrix
%  
%   Inputs:     height:         The height of the matrix
%               width:          The width of the matrix
%               r:              The size of the band
%  
%   Outputs:    band_mask:      A binary mask of dimensions height x width. 
%                               

slope = height / width;

is_inside = @(x, y) abs(y - slope * x) < r;

[ I, J ] = meshgrid(1 : width, 1 : height);
band_mask = bsxfun(is_inside, I, J);

end

