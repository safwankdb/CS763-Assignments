function [result_pose, composed_rot] = transformPose(rotations, pose, kinematic_chain, root_location)
    % rotations: A 15 x 3 x 3 array for rotation matrices of 15 bones
    % pose: The base pose coordinates 16 x 3.
    % kinematic chain: A 15 x 2 array of joint ordering
    % root_positoin: the index of the root in pose vector.
    % Your code here
    root_node.index = root_location;
    root_node.parent = root_location;
    root_node.position = pose(root_location, :);
    root_node.children = [];
    body_tree = [root_node];

    for i = 1:14
    	body_tree = [body_tree, root_node];
    end

    for i = 1:15
    	node.parent = kinematic_chain(i, 2);
    	node.index = kinematic_chain(i, 1);
    	node.position = pose(node.index, :) - pose(node.parent, :);
		node.children = [];
    	body_tree(node.parent).children = [body_tree(node.parent).children, node.index];
    	body_tree(node.index) = node;
    	clear('node');
    end

    % for i = 1:15
    % 	disp(body_tree(i))
    % end

    function rotated_joint = rotate(joint_index, rotation_matrix)
    	child_joint = joint_index;
    	bone = body_tree(child_joint).position;
    	rotated_bone = bone * rotation_matrix;
    	body_tree(child_joint).position = rotated_bone;

    	if(child_joint != root_location && size(body_tree(child_joint).children))
	    	for i = 1:size(body_tree(child_joint).children)
	    		body_tree(body_tree(child_joint).children(i)).position += rotated_bone - bone;
	    		rotate(body_tree(child_joint).children(i), rotation_matrix);
	    	end	
	   	end
    end

    for bone = 1:15
    	joint = kinematic_chain(bone, 1);
    	rotate(joint, squeeze(rotations(bone, :, :)))
    end

    result_pose = pose(:, :);
    for i = 1:16
    	result_pose(i, :) = body_tree(i).position;
    	j = i;
    	while(body_tree(j).parent != body_tree(j).index)
    		result_pose(i, :) += body_tree(body_tree(j).parent).position;
    		j = body_tree(j).parent;
    	end
    end
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