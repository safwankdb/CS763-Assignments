function JH = find_JH(im1,im2,binsize)

temp_im1 = im1(im2 ~=0 & im1 ~=0);
temp_im2 = im2(im2 ~=0 & im1 ~=0);
temp_im1 = ceil(temp_im1/10);
temp_im2 = ceil(temp_im2/10);

len = length(temp_im1);

numbins = ceil(255/binsize);
p12 = zeros(numbins);
for i=1:len
    p12(temp_im1(i),temp_im2(i)) = p12(temp_im1(i),temp_im2(i))+1;
end
p12 = p12/sum(p12(:));

JH = -sum(p12(p12 > 0).*log2(p12(p12 > 0)));