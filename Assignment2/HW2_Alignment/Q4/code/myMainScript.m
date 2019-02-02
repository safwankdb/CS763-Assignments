tic;

image_names = dir('../input/hill/*.JPG');

n = numel(image_names);
im_gray = rgb2gray(im.read(image_names(1)));
points = detectSURFFeatures();
[features, points] = extractFeatures(im_gray, points);
transforms = zeros(3, 3, n - 1);

for i = 2:n
   img_gray = rgb2gray(im.read(image_names(i)));
   previousPoints = points;
   previousFeatures = features;
   points = detectSURFFeatures(im_gray);
   [features, points] = extractFeatures(img_gray, points);
   
   indexPairs = matchFeatures
   
   homography = ransacHomography(previousPoints, currentPoint,  threshold);
   transforms(:, :, 1   ) = homography;
end

toc;
