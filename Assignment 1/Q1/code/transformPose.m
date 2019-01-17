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
    body_tree = [];

    for i = 1:15
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


    function rotate_joint = rotate(joint_index, rotation_matrix)
    	bone = body_tree(joint_index).position;
    	rotated_bone = bone * rotation_matrix;
    	body_tree(joint_index).position = rotated_bone;
    	if(joint_index != root_location)
    		children = body_tree(joint_index).children;
	    	for i = 1:size(children)
	    		child = children(i);
	    		rotate(child, rotation_matrix);
	    	end	
	   	end
    end

    for bone = 1:15
    	joint = kinematic_chain(bone, 1);
    	rotate(joint, squeeze(rotations(bone, :, :)))
    end

    result_pose = zeros(16, 3);
    for i = 1:16
    	result_pose(i, :) = body_tree(i).position;
    	parent_node = body_tree(body_tree(i).parent);
    	while(parent_node.parent != parent_node.index)
    		result_pose(i, :) += parent_node.position;
    		parent_node = body_tree(parent_node.parent);
    	end
    end
end
