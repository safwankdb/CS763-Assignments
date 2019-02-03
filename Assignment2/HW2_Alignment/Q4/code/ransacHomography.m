function [ H ] = ransacHomography( x1, x2, thresh )
%   RANSACHOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here
dataPoints = [x1, x2];
[H, ~] = ransac(dataPoints, @fitFunc, @dist, 4, thresh, 'Confidence', 99, 'MaxNumTrials', 2000);
end

