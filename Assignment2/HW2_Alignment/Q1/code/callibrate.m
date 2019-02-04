function [P]= callibrate(pts2d, pts3d)
N=size(pts2d,2);
M=zeros(2*N,12);
for i=1:N
    X=pts3d(1,i);
    Y=pts3d(2,i);
    Z=pts3d(3,i);
    x=pts2d(1,i);
    y=pts2d(2,i);
    M(2*i-1,:)=[-X -Y -Z -1 0 0 0 0 x*X x*Y x*Z x];
    M(2*i,:)=[0 0 0 0 -X -Y -Z -1 y*X y*Y y*Z y];
end
[U,S,V]=svd(M);
P=V(:,12)';
P=P./P(1,12);
P=reshape(P,[4,3])';