from BCART.ParallelBCART_BayesianTree import Node
import numpy as np
import scipy


def tree_proposal(row, random_numbers, X_train, y_train, min_split):

	tree_no = row[0]
	tree = row[1]

	alpha = tree.alpha
	beta = tree.beta

	which_step, rand_feature, rand_uniform1, rand_uniform2 = random_numbers.value[tree_no]

	# this is a flag to indicate is the tree has actually been change
	# for example, say the step is SWAP, but there isnt a pair of node
	# which can swap, then the tree remains unchanged
	flag = 0 
	
	if which_step == 0:
# *****************************
# **********   GROW    ********
# *****************************
		term_nodes = tree.terminal_nodes

		#select a node to grow
		chosen_node = int(round(rand_uniform1 * len(term_nodes)))
		if chosen_node == len(term_nodes):
			chosen_node = 0
		grow_node = term_nodes[chosen_node]
		
		# randomly select split feature             
		feature = rand_feature
		
		#randomly select split value
		feature_range = np.ptp(X_train.value[:,feature])
		feature_min = np.min(X_train.value[:,feature])
		rand_point = rand_uniform2 * feature_range
		threshold = feature_min + rand_point

		# check it is a valid split
		data_points = grow_node.data_indices
		all_leftData = np.where(X_train.value[:,feature] <= threshold)[0]
		leftData = np.intersect1d(all_leftData, data_points)

		all_rightData = np.where(X_train.value[:,feature] > threshold)[0]
		rightData = np.intersect1d(all_rightData, data_points)

		if leftData.size > min_split.value and rightData.size > min_split.value:
			#if it is a valid split >> create the children


			left_child = Node(isLeft = True, parent=grow_node, data_indices=leftData)
			right_child = Node(isLeft = False, parent=grow_node, data_indices=rightData)
			    
			grow_node.split_feature = feature
			grow_node.split_value = threshold

			tree.terminal_nodes.extend((left_child, right_child))
			tree.terminal_nodes.remove(grow_node)         
			tree.internal_nodes.append(grow_node)

			#update the qp_acceptance value
			prob_split = alpha * (1 + grow_node.depth)**-beta 

			parents = [x.parent for x in term_nodes]
			if len(parents) != 0:
			# 	possible_prunes = [x for x, y in collections.Counter(parents).items() if y == 2]

				unique_parents = set(parents)
				possible_prunes = []
				for node in unique_parents:
					if parents.count(node) >= 2:
						possible_prunes.append(node)
			
			no_possible_prunes = len(possible_prunes)
					
			no_terminal_nodes = len(term_nodes) 

			tree.qp_acceptance = (float(no_terminal_nodes)/float(no_possible_prunes))*(prob_split/(1-prob_split))

		else:
			flag = 1


	elif which_step == 1:
# *****************************
# ******    PRUNE    **********
# *****************************
		nodes = tree.terminal_nodes
		if len(nodes) != 0:
			parents = [x.parent for x in nodes]

			if len(parents) != 0:
				# possible_prunes = [x for x, y in collections.Counter(parents).items() if y == 2]
				unique_parents = set(parents)
				possible_prunes = []
				for node in unique_parents:
					if parents.count(node) >= 2:
						possible_prunes.append(node)

				no_possible_prunes = len(possible_prunes)
				
				if no_possible_prunes != 0:
					chosen_node = int(round(rand_uniform1 * no_possible_prunes))
					if chosen_node == no_possible_prunes:
						chosen_node = 0
					parent = possible_prunes[chosen_node]

					#update terminal and internal node lists
					tree.terminal_nodes.remove(parent.leftChild)
					tree.terminal_nodes.remove(parent.rightChild)
					tree.terminal_nodes.append(parent)  
					tree.internal_nodes.remove(parent)

					#update attributes
					parent.leftChild = None
					parent.rightChild = None
					parent.split_feature = None
					parent.split_value = None

					#update the qp_acceptance value
					prob_split = alpha * (1 + parent.depth)**-beta 
					no_terminal_nodes = len(tree.terminal_nodes)
					      
					tree.qp_acceptance=(float(no_possible_prunes)/float(no_terminal_nodes)*((1-prob_split)/prob_split))
		else:
			flag = 1

	elif which_step == 2:
# *****************************
# ******    CHANGE    *********
# *****************************
		inter_nodes = tree.internal_nodes
		
		if len(inter_nodes) != 0:
			chosen_node = int(round(rand_uniform1 * len(inter_nodes)))
			if chosen_node == len(inter_nodes):
				chosen_node = 0
			c_node = inter_nodes[chosen_node]

			#save old value incase it is a invalid split
			old_split_value = c_node.split_value
			feature = c_node.split_feature

			#generate a new split value
			feature_range = np.ptp(X_train.value[:,feature])
			feature_min = np.min(X_train.value[:,feature])
			rand_point = rand_uniform2 * feature_range
			threshold = feature_min + rand_point

			c_node.split_value = threshold
			reset_nodes = [c_node.leftChild, c_node.rightChild]

			for node in reset_nodes:
				parent = node.parent

				if node.isLeft == True:
				
					all_Data = np.where(X_train.value[:, parent.split_feature] <= parent.split_value)[0]
				
				else:

					all_Data = np.where(X_train.value[:, parent.split_feature] > parent.split_value)[0]

				data_points = np.intersect1d(all_Data, parent.data_indices)

				if data_points.size <= min_split.value:
					flag = 1

				else:
					node.data_indices = data_points

					if node.leftChild is not None:
						reset_nodes.append(node.leftChild)
						reset_nodes.append(node.rightChild)

			if flag == 1:
				c_node.split_value = old_split_value

				reset_nodes = [c_node.leftChild, c_node.rightChild]

				for node in reset_nodes:
					parent = node.parent

					if node.isLeft == True:
				
						all_Data = np.where(X_train.value[:, parent.split_feature] <= parent.split_value)[0]
					
					else:

						all_Data = np.where(X_train.value[:, parent.split_feature] > parent.split_value)[0]

					data_points = np.intersect1d(all_Data, parent.data_indices)
					
					node.data_indices = data_points

					if node.leftChild is not None:
						reset_nodes.append(node.leftChild)
						reset_nodes.append(node.rightChild)
			else:
				tree.qp_acceptance = 1

	elif which_step == 3:

# *****************************
# ********    SWAP    *********
# *****************************
		#get list of nodes which can be swapped (both internal)            
		possible_swaps = tree.internal_nodes

		#if there is a swapping pair...
		if len(possible_swaps) > 1:
			
			#remove root not from the list
			if possible_swaps[0] == tree.root_node:
				possible_swaps = possible_swaps[1:]
			
			#select the swapping node (this is the child in the pair)
			chosen_node = int(round(rand_uniform1 * len(possible_swaps)))
			if chosen_node == len(possible_swaps):
				chosen_node = 0
			swap_node = possible_swaps[chosen_node]

			#set the swap parent
			swap_parent = swap_node.parent

			#hold the values from the swap child
			temp_feature = swap_node.split_feature
			temp_value = swap_node.split_value

			#assign the new values for the child
			swap_node.split_feature = swap_parent.split_feature
			swap_node.split_value = swap_parent.split_value

			#assign the new values for the parent       
			swap_parent.split_feature = temp_feature
			swap_parent.split_value = temp_value

			reset_nodes = [swap_parent.leftChild, swap_parent.rightChild]
		
			for node in reset_nodes:
				parent = node.parent
				
				if node.isLeft == True:
				
					all_Data = np.where(X_train.value[:, parent.split_feature] <= parent.split_value)[0]
				
				else:

					all_Data = np.where(X_train.value[:, parent.split_feature] > parent.split_value)[0]

				data_points = np.intersect1d(all_Data, parent.data_indices)

				if data_points.size <= min_split.value:
					flag = 1
					# break

				else:
					node.data_indices = data_points

					if node.leftChild is not None:
						reset_nodes.append(node.leftChild)
						reset_nodes.append(node.rightChild)

			if flag == 1:
				#hold the values from the swap child
				temp_feature = swap_node.split_feature
				temp_value = swap_node.split_value

				#assign the new values for the child
				swap_node.split_feature = swap_parent.split_feature
				swap_node.split_value = swap_parent.split_value

				#assign the new values for the parent       
				swap_parent.split_feature = temp_feature
				swap_parent.split_value = temp_value

				reset_nodes = [swap_parent.leftChild, swap_parent.rightChild]

				for node in reset_nodes:
					node_parent = node.parent
					
					if node.isLeft == True:
				
						all_Data = np.where(X_train.value[:, node_parent.split_feature] <= node_parent.split_value)[0]
					
					else:

						all_Data = np.where(X_train.value[:, node_parent.split_feature] > node_parent.split_value)[0]

					data_points = np.intersect1d(all_Data, node_parent.data_indices)

					node.data_indices = data_points

					if node.leftChild is not None:
						reset_nodes.append(node.leftChild)
						reset_nodes.append(node.rightChild)
			else:
				tree.qp_acceptance = 1


# ******************************
# ***   update the weight    ***
# ** if the tree is unchanged **
# ******************************
	if flag == 0:
		# tree.flag = 0
		old_weight = row[2]
		accp = tree.qp_acceptance
		lik_pre = tree.likelihood
		# tree.weight_pre = tree.weight

		likelihood = 1

		for i in tree.terminal_nodes:
			node_indices = i.data_indices
			labels = y_train.value[node_indices]

			counts = np.bincount(labels)
			i.class_label = np.argmax(counts)

			if counts.shape[0] == 1:
				n_0 = counts[0]
				n_1 = 0

			else:
				n_0 = counts[0]
				n_1 = counts[1]

			beta_fun = scipy.special.beta(1 + n_0, 1 + n_1)
			likelihood *= beta_fun

		
		# new_weight =  old_weight * accp * likelihood / lik_pre
		new_weight = old_weight * accp * likelihood / lik_pre
		tree.likelihood = likelihood
		
		row = (tree_no, tree, new_weight)


	return row

