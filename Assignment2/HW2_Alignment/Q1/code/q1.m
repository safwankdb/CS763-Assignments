% Check normalise2D
original2d=[0 1
    1 0
    1 1];
[pts2d,T]=normalise2D(original2d)

% Check normalise3D
original3d=[0 1
    1 0
    0 0 
    1 1];
[pts3d,U]=normalise3D(original3d);

[P]=callibrate(pts2d,pts3d);
P=inv(T)*P*U;
P=P./P(3,4);

%Checking callibrate function
image=P*original3d;
image(:,1)=image(:,1)./image(3,1) ;
image(:,2)=image(:,2)./image(3,2);
disp(image)
%correct since image=original2d
