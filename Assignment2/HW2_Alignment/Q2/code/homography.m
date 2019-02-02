function [ H ] = homography( p1, p2 )
%   HOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here
    assert(isequal(size(p1), size(p2)), 'Different Number of Points')
    n = size(p1, 1);
    A = zeros( 2*n,9);
    for i = 1:n
        a = [ -p1(i, :), zeros(1, 3), p1(i, :) * p2(i, 1) ];
        b = [ zeros(1, 3), -p1(i, :), p1(i, :) * p2(i, 2) ];
        j = 2*i;
        A(j-1:j, :) = [ a; b ];
    end
    [~, ~, V] = svd(A);
    H = V(:, end);
    H = reshape(H, 3, 3);
end