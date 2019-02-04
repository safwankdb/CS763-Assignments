clear;

tic;

threshold = 0.01;
image_dir = '../input/pier';
image_names = dir(fullfile(image_dir, '*.JPG'));

n = numel(image_names);
img = imread(fullfile(image_dir, image_names(1).name));
img_gray = rgb2gray(img);
points = detectSURFFeatures(img_gray, 'MetricThreshold', 1000);
[features, points] = extractFeatures(img_gray, points);
transforms(n) = projective2d(eye(3));

image_sizes = zeros(n, 2);
image_sizes(1,:) = size(img_gray);

for i = 2:n
   img_prev = img;
   img = imread(fullfile(image_dir, image_names(i).name));
   
   img_gray = rgb2gray(img);
   image_sizes(i, :) = size(img_gray);
   
   previousPoints = points;
   previousFeatures = features;
   
   points = detectSURFFeatures(img_gray, 'MetricThreshold', 1000);
   [features, points] = extractFeatures(img_gray, points);
      
   
   indexPairs = matchFeatures(features, previousFeatures, ...
  'Unique', true);
%    'MatchThreshold', 10, 'MaxRatio', 0.6, ...
%    'Method', 'Approximate');
   
   matchedPoints = points(indexPairs(:, 1));
   matchedPointsPrev = previousPoints(indexPairs(:, 2));

%    
%    figure
%    showMatchedFeatures(img_prev, img, matchedPointsPrev, matchedPoints);
   
   previousLocation = [matchedPointsPrev.Location, ones(size(matchedPointsPrev, 1), 1)];
   location = [matchedPoints.Location, ones(size(matchedPoints, 1), 1)];
      
   H_transform = ransacHomography(location, previousLocation,  threshold);
   transforms(i) = projective2d(H_transform).T * transforms(i-1).T ;
end

for i = 1:n
    [x_limit(i, :), y_limit(i, :)] = outputLimits(transforms(i),...
        [1, image_sizes(i,2)], [1, image_sizes(i,1)]);
end

avgXLim = mean(x_limit, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((n+1)/2);

centerImageIdx = idx(centerIdx);

Tinv = invert(transforms(centerImageIdx));

for i = 1:n
    transforms(i).T = transforms(i).T * Tinv.T;
end

for i = 1:numel(transforms)
    [x_limit(i,:), y_limit(i,:)] = outputLimits(transforms(i), [1 image_sizes(i,2)], [1 image_sizes(i,1)]);
end

maxImageSize = max(image_sizes);
xMin = min([1; x_limit(:)]);
xMax = max([maxImageSize(2); x_limit(:)]);

yMin = min([1; y_limit(:)]);
yMax = max([maxImageSize(1); y_limit(:)]);

width  = round(xMax - xMin);
height = round(yMax - yMin);

panorama = zeros([height width 3], 'like', img);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:n
   I = imread(fullfile(image_dir, image_names(i).name));
   warpedImage = imwarp(I, transforms(i), 'OutputView', panoramaView);
   % Generate a binary mask.
   mask = imwarp(true(size(I,1),size(I,2)), transforms(i), 'OutputView', panoramaView);
    % Overlay the warpedImage onto the panorama.
   panorama = blender(panorama, warpedImage, mask);
end


imshow(panorama);

toc;
