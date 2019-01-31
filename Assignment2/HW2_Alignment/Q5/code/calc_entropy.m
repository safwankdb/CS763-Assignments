function entro_ans = calc_entropy(im1,im2)
t1_1=im1(im1~=0 & im2~=0);
t2_1=im2(im1~=0 & im2~=0);
t1_2=floor(t1_1./10)+1;
t2_2=floor(t2_1./10)+1;
hist=zeros(26,26);
len=length(t1_1);
entro_ans=0;
for i=1:len
    hist(t1_2(i),t2_2(i))=hist(t1_2(i),t2_2(i))+1;
end
hist=hist/sum(hist(:));
new_mat=hist(hist>0).*log2(hist(hist>0));
entro_ans=-sum(new_mat(:));