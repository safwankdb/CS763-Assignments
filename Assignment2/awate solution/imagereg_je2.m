clear; close; clc;
im1 = double((imread('barbara.png'))); im1 = im1+1;
im2 = double((imread('negative_barbara.png'))); im2 = im2+1;
im2(1:128,1:128) = im2(1:128,1:128).^0.25;
im2(1:128,129:256) = im2(1:128,129:256).^0.33;
im2(129:256,1:128) = sqrt(im2(129:256,1:128));
im2(129:256,129:256) = im2(129:256,129:256).^0.33 + sqrt(im2(129:256,129:256));
im2 = 255*(im2-min(im2(:)))/(max(im2(:))-min(im2(:)))+1;

temp_im2 = imrotate(im2,23.5,'bilinear','crop');
[H,W] = size(temp_im2);
im2 = zeros(H,W);

im2(:,1:W-3) = temp_im2(:,4:W);

%im2(:,7:W) = temp_im2(:,1:W-6);
% im2(:,1:W) = temp_im2(:,1:W);

im2 = im2 + randn(size(im2))*8;
imshow(im2/max(im2(:)));

thetas = -60:1:60;
txs = -12:1:12;
binsize = 10;

count_theta = 0;
for theta=-60:1:60
     fprintf ('%d ',theta);
    
     temp_im2 = imrotate(im2,theta,'bilinear','crop');
     
     count_theta = count_theta+1;
     count_tx = 0;
     
     for tx=-12:1:12
        count_tx = count_tx + 1; 
        temp_im3 = zeros(H,W);
        if tx > 0,
            temp_im3(:,tx+1:W) = temp_im2(:,1:W-tx);
        elseif tx < 0,
            temp_im3(:,1:W+tx) = temp_im2(:,-tx+1:W);
        else
            temp_im3 = temp_im2;            
        end
     
        JH(count_theta,count_tx) = find_JH(im1,temp_im3,binsize);
     end
 end

figure, surf(JH);
figure, imshow(JH/max(JH(:)));
minval = min(JH(:));
[theta_index,tx_index] = find(abs(JH-minval) <= 0.00001);
fprintf ('\nMin JH = %f, occurs at theta = %d, tx = %d',minval,thetas(theta_index),txs(tx_index));