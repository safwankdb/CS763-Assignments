close all
clc
clear
im1 = double(imread('../input/barbara.png'))./255;
im2 = double(imread('../input/negative_barbara.png'))./255;
tic;
angle = 60;
space = 12;


[x,y]=size(im1)

im2_1=imrotate(im2,23.5,'bilinear','crop');
im2_2=imtranslate(im2_1,[-3,0]);
noise=(8.*rand(size(im2)))./255;
noise(im2_2<0.00001)=0;
im2_3=min(1,max(0,im2_2+noise));
im2_4=im2_3*255;

im1_1=im1*255;
entropy=zeros(2*space+1,2*angle +1);
min_val=100000;
min_theta=0; min_t=0;
fprintf('Running for rotation angle\n');
for theta=-angle:1:angle
    fprintf('%d\n',theta);
    temp = imrotate(im2_4,theta,'bilinear','crop');
    for t = -space:1:space
        temp_1 = imtranslate(temp,[t,0]);
%         figure; imshow(temp_1/255);
        an = calc_entropy(temp_1,im1_1);
        entropy(t+space + 1,theta+angle+1) = an;
        if an<min_val
            min_val=an;
            min_t=t;
            min_theta=theta;
        end
    end
end

[angles,tr]=meshgrid(-angle:1:angle,-space:1:space);
surf(angles,tr,entropy);

figure,imshow((entropy-min(entropy(:)))/(max(entropy(:))-min(entropy(:))));
fprintf ('\nMin joint entropy value %d occurs at theta = %d, tx = %d\n',min_val,min_theta,min_t);
toc;
