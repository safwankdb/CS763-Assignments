clear;
clc;
points2d = load('../input/xy.mat');
points2d = reshape(points2d.A, [], 2);
points2d = [points2d, ones(size(points2d, 1), 1)]';

points3d = load('../input/xyz.mat');
points3d = reshape(points3d.B, [], 3);
points3d = [points3d, ones(size(points3d, 1), 1)]';

[points2dNew, T] = normalise2D(points2d);

[points3dNew, U] = normalise3D(points3d);

P = calibrate(points2dNew, points3dNew);
P = (T \ P) * U;
P = P ./ P(3, 4)

H_inf = P(:, 1:3);
h = P(:, 4);

X_o = -(H_inf \ h);
[R_t, K_i] = qr(inv(H_inf));
R = R_t'
K = inv(K_i)

pad = ones(3, 1);
imagePoints = P * points3d;
imagePoints = imagePoints ./ (pad * imagePoints(3, :));

MSE = mean((imagePoints(:) - points2d(:)).^2) * 3;
RMSE = sqrt(MSE)

figure;
imshow(imread('../input/IMG.jpeg'))
hold on
plot(points2d(1, :)', points2d(2, :)', 'yx', 'MarkerSize', 10)
plot(imagePoints(1, :)', imagePoints(2, :)', 'r+', 'MarkerSize',8)