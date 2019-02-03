function [ H ] = homography( data )
%   HOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here

    p1 = data(:, 1:3);
    p2 = data(:, 4:6);
    n = size(data, 1);
    A = zeros( 2*n,9);
    for i = 1:n
        a = [ -p1(i, :), zeros(1, 3), p1(i, :) * p2(i, 1) ];
        b = [ zeros(1, 3), -p1(i, :), p1(i, :) * p2(i, 2) ];
        A(2*i-1: 2*i, :) = [ a; b ];
    end
    [~, ~, V] = svd(A); 
    H = V(:, end);
    H = reshape(H, 3, 3);
    H = H/H(3, 3);
%     H = H'
end
