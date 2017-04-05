function [ feat ] = hog( im, cellsize )
%
% input: im: image
%        cs: cell size
%
% output: feat: hog feature representation of the image

nbins = 9;

[im2,h,w] = set_imsize(im,cellsize);

dx = conv2( im2, [-1 0 1], 'valid' ); % image gradient x
dx = [ zeros(h*cellsize,1), dx, zeros(h*cellsize,1) ];
dy = conv2( im2, [-1 0 1]', 'valid' ); % image gradient y
dy = [ zeros(1,w*cellsize); dy; zeros(1,w*cellsize) ];

ang = atan2(dy,dx) + pi; % orientation
r = sqrt( dx.^2 + dy.^2 ); % length
r_cells = im2col_distinct( r, [cellsize cellsize] );
ang_cells = im2col_distinct( ang, [cellsize cellsize] );

tmp = zeros( 2*nbins+1, h*w );
for i = 0:2*nbins % linear interpolation for bin assignment
    membership = max( 1-abs(i*pi/nbins - ang_cells)*nbins/pi, 0 );
    tmp(i+1,:) = sum( r_cells .* membership, 1 );
end
rawhog = tmp(2:2*nbins+1,:);
rawhog(2*nbins,:) = rawhog(2*nbins,:) + tmp(1,:);
hd = reshape( rawhog', [h,w,2*nbins] );
hu = hd(:,:,1:nbins) + hd(:,:,nbins+1:2*nbins);

[ hd, ~ ] = normalize22(hd);
[ hu, l1 ] = normalize22(hu);
feat = cat( 3, hd, hu, l1 );

end

function [im2,h,w] = set_imsize(im,cellsize)

s = [ size(im,1), size(im,2) ];
s2 = round( s / cellsize ) * cellsize;
im2 = zeros( [s2(1), s2(2)] );
smin = min(s,s2);
im2( 1:smin(1), 1:smin(2) ) = im( 1:smin(1), 1:smin(2) );
h = s2(1) / cellsize;
w = s2(2) / cellsize;

end

function [ hn, l1 ] = normalize22(h)

s = size(h);
tmp = conv2( sum( h.^2, 3 ), [1 1; 1 1], 'valid' );
tmp = [ tmp(1,:); tmp; tmp(end,:) ];
tmp = [ tmp(:,1), tmp, tmp(:,end) ];
l22 = im2col_sliding( tmp, [2 2] );
l2 = reshape( sqrt(l22'), [s(1),s(2),4] );

hn = cell(1,4); l1 = cell(1,4);
for i = 1:size(l2,3)
    hn{i} = bsxfun(@rdivide,h,l2(:,:,i));
    hn{i}(isnan(hn{i})) = 0;
    hn{i} = min( hn{i}, 0.2 );
    l1{i} = sum( abs(hn{i}), 3 );
end
hn = sum( cat(4,hn{:}), 4 );
l1 = cat( 3, l1{:} );

end
