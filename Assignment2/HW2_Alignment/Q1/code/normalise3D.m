function [pts_out, T]=normalise3D(pts_in) %input is a 4*N matrix of points
old_pts=pts_in;
N=size(pts_in,2);
pts_in(1,:)=pts_in(1,:)./pts_in(4,:);
pts_in(2,:)=pts_in(2,:)./pts_in(4,:);
pts_in(3,:)=pts_in(3,:)./pts_in(4,:);
pts_in(4,:)=ones(1,N);
mean_v=zeros(3,1);
mean_v(1,1)=mean(pts_in(1,:));
mean_v(2,1)=mean(pts_in(2,:));
mean_v(3,1)=mean(pts_in(3,:));
pts_in(1,:)=pts_in(1,:)-mean_v(1,1);
pts_in(2,:)=pts_in(2,:)-mean_v(2,1);
pts_in(3,:)=pts_in(3,:)-mean_v(3,1);
% disp(pts_in)
x=pts_in(1,:);
y=pts_in(2,:);
z=pts_in(3,:);
fact=sqrt(x.*x+y.*y+z.*z);
mean_f=mean(fact);
mean_f=sqrt(3)/mean_f ;
% disp(mean_f)
T=[mean_f 0 0 -mean_f*mean_v(1,1)
    0 mean_f 0 -mean_f*mean_v(2,1)
    0  0 mean_f -mean_f*mean_v(3,1)
    0 0 0 1];
pts_out = T*old_pts;