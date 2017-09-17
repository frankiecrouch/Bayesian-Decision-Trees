import numpy as np
from sklearn import metrics



def calc_auc(row, y_train, X_test, y_test):

	tree = row[1]

	# ******
	# TRAIN
	# ******
	y_predicted_train = np.zeros(y_train.value.shape[0])
	
	# set the class of each terminal node
	for node in tree.terminal_nodes:
		   
	   if node.class_label == 1:
		   np.put(y_predicted_train, node_indices, 1)

	auc_train = metrics.roc_auc_score(y_train.value, y_predicted_train)

	# ******
	# TEST
	# ******

	# get the data indices of the test data for each terminal node
	if tree.root_node.leftChild is not None:
		
		set_test_indicies = [tree.root_node.leftChild, tree.root_node.rightChild]
	   
		for node in set_test_indicies:
			
			parent = node.parent

			if node.isLeft == True:
			
				all_data = np.where(X_test.value[:, parent.split_feature] <= parent.split_value)[0]
			
			else:
				
				all_data = np.where(X_test.value[:, parent.split_feature] > parent.split_value)[0]
			
			data = np.intersect1d(all_data, parent.test_data_indices)

			node.test_data_indices = data

			if node.leftChild is not None:
				set_test_indicies.append(node.leftChild)
				set_test_indicies.append(node.rightChild)
	
	# calculate the AUC
	y_predicted = np.zeros(y_test.value.shape[0])

	for term_node in tree.terminal_nodes:

		if term_node.class_label == 1:
		   node_indices = term_node.test_data_indices
		   np.put(y_predicted, node_indices, 1)
	
	auc = metrics.roc_auc_score(y_test.value, y_predicted)

	row = (auc, auc_train)

	
	return row


