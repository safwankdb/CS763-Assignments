function entro_ans = calc_entropy(im1,im2)
t1_1 = im1(im1~=0 & im2~=0);
t2_1 = im2(im1~=0 & im2~=0);
t1_2 = ceil(t1_1 / 10);
t2_2 = ceil(t2_1 / 10);

len = length(t1_1);
hist = zeros(26);
entro_ans=0;
for i=1:len
    hist(t1_2(i),t2_2(i)) = hist(t1_2(i),t2_2(i))+1;
end
hist=hist/sum(hist(:));
entro_ans=-sum(hist(hist>0).*log2(hist(hist>0)));
