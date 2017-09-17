from BCART.ParallelBCART_BayesianTree import Node
import numpy as np
import scipy

def build_tree(row, random_numbers, X_train, y_train, min_split):
    
    tree_no = row[0]
    tree = row[1]
    alpha = tree.alpha
    beta = tree.beta


    # ************************
    # First build to the tree
    # ************************
    
    # get the random numbers
    rand_numbers = random_numbers.value[tree_no]
    
    # creat list to iterate over
    split_nodes = [tree.root_node]    

    
    k = -1
    for node in split_nodes:
        k += 1
        if k == 20:
            k=0
        
        # decide whether to split the current noe
        current_node = node       
        prob_of_split = alpha * (1 + current_node.depth)**-beta
        
        rand_accept = rand_numbers[k][0]
        
        if rand_accept < prob_of_split:
            
            #randomly select split feature             
            feature = rand_numbers[k][1]

            #randomly select split value
            feature_range = np.ptp(X_train.value[:,feature])
            feature_min = np.min(X_train.value[:,feature])
            rand_point = rand_numbers[k][2] * feature_range
            threshold = feature_min + rand_point
            
            #check it is a valid split
            all_leftData = np.where(X_train.value[:,feature] <= threshold)[0]
            leftData = np.intersect1d(all_leftData, current_node.data_indices)

            all_rightData = np.where(X_train.value[:,feature] > threshold)[0]
            rightData = np.intersect1d(all_rightData, current_node.data_indices)
            
            if leftData.size > min_split.value and rightData.size > min_split.value:
                
                #if it is a valid split >> create the children
                left_child = Node(isLeft = True, parent=current_node, data_indices=leftData)
                right_child = Node(isLeft = False, parent=current_node, data_indices=rightData)
                
                current_node.split_feature = feature
                current_node.split_value = threshold
                
                tree.terminal_nodes.extend((left_child, right_child))
                tree.terminal_nodes.remove(current_node)         
                tree.internal_nodes.append(current_node)

                # add the children to the list - to see if they are split
                split_nodes.append(left_child)
                split_nodes.append(right_child)
                
    # *****************************************
    # Calculate the starting weight
    # q(t) = p(t), so it is just the likelihood
    # *****************************************

    likelihood = 1

    for i in tree.terminal_nodes:
        node_indices = i.data_indices
        labels = y_train.value[node_indices]

        counts = np.bincount(labels)

        if counts.shape[0] == 1:
            n_0 = counts[0]
            n_1 = 0

        else:
            n_0 = counts[0]
            n_1 = counts[1]
        
        beta_fun = scipy.special.beta(1 + n_0, 1 + n_1)
        likelihood *= beta_fun
    
    tree.likelihood = likelihood
    
    row = (tree_no, tree, likelihood)

    return row