function [pts_out, T] = normalise2D(pts_in)
%input is a 3*N matrix of points

old_pts = pts_in;
N = size(pts_in, 2);

pts_in(1, :) = pts_in(1, :) ./ pts_in(3, :);
pts_in(2, :) = pts_in(2, :) ./ pts_in(3, :);
pts_in(3, :) = ones(1, N);

mean_v = zeros(2, 1);
mean_v(1, 1) = mean(pts_in(1, :));
mean_v(2, 1) = mean(pts_in(2, :));

pts_in(1, :) = pts_in(1, :) - mean_v(1, 1);
pts_in(2, :) = pts_in(2, :) - mean_v(2, 1);

% disp(pts_in)
x = pts_in(1, :);
y = pts_in(2, :);
fact = sqrt(x.^2 + y.^2);
mean_f = mean(fact);
mean_f = sqrt(2) / mean_f ;
% disp(mean_f)

T=[ 
    mean_f 0      -mean_f*mean_v(1,1);
    0      mean_f -mean_f*mean_v(2,1);
    0      0       1
   ];
% size(old_pts)
pts_out = T * old_pts;