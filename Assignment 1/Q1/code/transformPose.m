function [result_pose, composed_rot] = transformPose(rotations, pose, kinematic_chain, root_location)
    % rotations: A 15 x 3 x 3 array for rotation matrices of 15 bones
    % pose: The base pose coordinates 16 x 3.
    % kinematic chain: A 15 x 2 array of joint ordering
    % root_positoin: the index of the root in pose vector.
    % Your code here
    function rotated_joint = rotate(child_joint, parent_joint, rotation)














	% bone_vector = pose(kinematic_chain(:,1),:) - pose(kinematic_chain(:,2),:);
	% rotated_bone = zeros(15, 3);
	% for i = 1:15
	% 	rotated_bone(i,:) = bone_vector(i,:) * transpose(squeeze(rotations(i,:,:)));
	% end
	% rotated_pose = rotated_bone + pose(kinematic_chain(:, 2), :);
	% result_pose = [
	% 				rotated_pose(1:root_location-1, :);
	% 				[0, 0, 0];
	% 				rotated_pose(root_location:15, :)
	% 				];
end