function [ resized ] = fake_imresize( image, new_size )
% FAKE_IMRESIZE Use this if you don't have 'imresize' from the image
% processing toolbox.
%
%   Inputs:         image:      A 2D matrix (no color channels)
%                   new_size:   Vector containing desired height and width
%   
%   Outputs:        resized:    The resized image


h = size(image, 1);
w = size(image, 2);

h_new = new_size(1);
w_new = new_size(2);

[ Xq, Yq ] = meshgrid(linspace(1, w, w_new), linspace(1, h, h_new));

% Use nearest interpolation because images are binarized anyways
resized = interp2(double(image), Xq, Yq, 'nearest');

end

