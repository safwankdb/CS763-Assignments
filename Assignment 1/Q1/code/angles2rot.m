function rot_matrix = angles2rot(angles_list)
    %% Your code here
    % angles_list: [theta1, theta2, theta3] about the x,y and z axes,
    % respectively.
    rot_matrix = zeros(15,3,3);
    for i = 1:15
    	[theta1, theta2, theta3] = num2cell(angles_list(i,:)){:};
	    R_x = [
	    		1 0 0;
	    	    0 cosd(theta1) -sind(theta1);
	    	    0 sind(theta1) cosd(theta1)
	    	   ];

		R_y = [
				cosd(theta2) 0 sind(theta2);
			    0 1 0;
			    -sind(theta2) 0 cosd(theta2)
			   ];

		R_z = [
				cosd(theta3) -sind(theta3) 0;
				sind(theta3) cosd(theta3)  0;
				0 0 1
			   ];
	    R = R_z * R_y * R_x;
	    rot_matrix(i,:,:) = R;
	end
end
