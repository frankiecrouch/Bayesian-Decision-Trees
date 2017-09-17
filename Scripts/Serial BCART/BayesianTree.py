from __future__ import division
import numpy as np
from graphviz import Digraph
import collections
from sklearn import metrics
from scipy.special import beta
import sys

# update the min_split value *****
min_split = 10


#****************************************************************************************************
# NODE CLASS
#****************************************************************************************************
class Node(object):

    nodeId = 0

    def __init__(self,isLeft=None,parent=None,split_feature=None,split_value=None, data_indices = None):

        self.id = Node.nodeId
        Node.nodeId += 1

        self.isLeft = isLeft
        self.parent = parent
        self.leftChild = None
        self.rightChild = None

        if self.parent is not None:
            if self.isLeft:
                parent.leftChild = self

            else:
                parent.rightChild = self
            
            self.depth = self.parent.depth + 1
        
        else:
            self.isLeft = None
            self.depth = 0

        self.data_indices = data_indices
        self.split_feature = split_feature
        self.split_value = split_value
        
        self.class_label = None
        self.test_data_indices = None
        


#****************************************************************************************************
# TREE CLASS
#****************************************************************************************************

class Tree(object):

    def __init__(self, root_node, alpha, beta, X, Y):
        
        self.root_node = root_node
        self.terminal_nodes = [root_node]
        self.internal_nodes = []
        
        self.alpha = alpha
        self.beta = beta
        
        self.X = X
        self.Y = Y
        
        self.likelihood = 0
        self.qp_acceptance = 1

    
    def calc_likelihood(self):


        likelihood = 1
        
        for i in self.terminal_nodes:
            node_indices = i.data_indices
            labels = self.Y[node_indices]
            
            counts = np.bincount(labels)

            
            if counts.shape[0] == 1:
                n_0 = counts[0]
                n_1 = 0
                
            else:
                n_0 = counts[0]
                n_1 = counts[1]
                       
            likelihood *= beta(1 + n_0, 1 + n_1)
            
            # update the class label of the node
            i.class_label = np.argmax(counts) 
 
              
        self.likelihood = likelihood

# ****** GROW ***********
    def grow(self):
        
        #randomly select one of the terminal nodes to grow
        random_number = np.random.randint(len(self.terminal_nodes))
        g_node = self.terminal_nodes[random_number]
      
        #create the left and right child
        left_child, right_child = self.create_children(g_node)
            
        if left_child is not None:
            #update the terminal node lists
            self.terminal_nodes.extend((left_child, right_child))
            self.terminal_nodes.remove(g_node)         
            self.internal_nodes.append(g_node)

            #update the qp_acceptance value
            prob_split = self.psplit(g_node)  
            no_possible_prunes = len(self.prune_list())
            no_terminal_nodes = len(self.terminal_nodes) - 1
            
            self.qp_acceptance = (float(no_terminal_nodes)/float(no_possible_prunes))*(prob_split/(1-prob_split))

    
    def create_children(self, parent):
        
        if parent.leftChild is not None or parent.rightChild is not None:
            return None, None
          
        #get a split value and feature for the parent node
        feature, threshold = self.prule(parent)

        #work out the data indices for each child  
        leftData, rightData = self.split_data(parent,feature, threshold)
        
        #check if still have min number of instances in each child
        if leftData.size <= min_split or rightData.size <= min_split:
            return None, None
        
        #create new nodes
        nleft = Node(isLeft = True,parent=parent, data_indices=leftData)
        nright = Node(isLeft = False,parent=parent, data_indices=rightData)
       
       #assign feature and threshold to parent
        parent.split_feature = feature
        parent.split_value = threshold
        
        return nleft, nright
        
    def prule(self, node, feature = None):
        #getting the FEATURE
        #for the grow step >> need to assign the feature
        if feature is None:
            feature = np.random.randint(self.X.shape[1]) 
        
        #for the swap step the feature remains unchanged
        else:
            feature = feature

        #getting the THRESHOLD  
        feature_range = np.ptp(self.X[:,feature])

        feature_min = np.min(self.X[:,feature])

        random_point = np.random.uniform(0,feature_range)

        threshold = feature_min + random_point
 
        return feature, threshold
        
    
    def split_data(self, parent, feature, threshold):
         
        all_leftData = np.where(self.X[:,feature] <= threshold)[0]
        leftData = np.intersect1d(all_leftData, parent.data_indices)
        
        all_rightData = np.where(self.X[:,feature] > threshold)[0]
        rightData = np.intersect1d(all_rightData, parent.data_indices)
       
        return leftData, rightData
        
        
    def psplit(self, node):
        prob_of_split = self.alpha * (1 + node.depth)**-self.beta
        
        return prob_of_split
        
    def prune_list(self):
        nodes = self.terminal_nodes
        if len(nodes) == 0:
            return
        parents = [x.parent for x in nodes]
        if len(parents) == 0:
            return
        # make sure there are 2 terminal children
        tparents = [x for x, y in collections.Counter(parents).items() if y == 2]
        
        return tparents
        
# ****** PRUNE ***********      
    def prune(self):
        
        #find the number of nodes which could be pruned
        possible_prunes = self.prune_list()
        no_possible_prunes = len(possible_prunes)
       
        if no_possible_prunes == 0:
            return None
        
        #select the node the prune
        parent  = possible_prunes[np.random.randint(no_possible_prunes)]
        
        #update terminal and internal node lists
        self.terminal_nodes.remove(parent.leftChild)
        self.terminal_nodes.remove(parent.rightChild)
        self.terminal_nodes.append(parent)  
        self.internal_nodes.remove(parent)
        
        #update attributes
        parent.leftChild = None
        parent.rightChild = None
        parent.split_feature = None
        parent.split_value = None

        #update the qp_acceptance value
        prob_split = self.psplit(parent)  
        no_terminal_nodes = len(self.terminal_nodes) 
              
        self.qp_acceptance=(float(no_possible_prunes)/float(no_terminal_nodes)*((1-prob_split)/prob_split))
    
      
# ****** CHANGE ***********       
    def change(self):
        
        #select node to change split value
        no_internal_nodes = len(self.internal_nodes)
        
        if no_internal_nodes > 0:
            random_number = np.random.randint(no_internal_nodes)
            c_node = self.internal_nodes[random_number]
           
            #save old value incase it is a invalid split
            old_split_value = c_node.split_value
            
            #generate a new split value
            c_node.split_feature, c_node.split_value = self.prule(c_node, c_node.split_feature)
            
            #reset the data indices for all nodes below the change node
            self.reset_indices(c_node, c_node.leftChild, c_node.rightChild)

            #change this is a valid split
            for i in self.terminal_nodes:
                #if the no of instances in a terminal node is below the threshold
                #switch back to the original split value and reset data indices
                if i.data_indices.size <= min_split:
                    c_node.split_value = old_split_value
                    self.reset_indices(c_node, c_node.leftChild, c_node.rightChild)
                    break
                #if it is a valid split, set the acceptance to 1                
                else:
                    self.qp_acceptance = 1

    def reset_indices(self, parent, left, right):
        
        leftData,rightData = self.split_data(parent,parent.split_feature, parent.split_value)
        
        left.data_indices = leftData
        right.data_indices = rightData

        
        if left.split_value is None and right.split_value is None:
            return
        
        elif left.split_value is not None and right.split_value is None:
            return self.reset_indices(left, left.leftChild, left.rightChild)
        
        elif left.split_value is None and right.split_value is not None:
            return self.reset_indices(right, right.leftChild, right.rightChild)
        
        else:
            return self.reset_indices(left, left.leftChild, left.rightChild), self.reset_indices(right, right.leftChild, right.rightChild)


# ****** SWAP ***********         
    def swap(self):
        
        #get list of nodes which can be swapped (both internal)            
        possible_swaps = self.internal_nodes
  
        #if there is a swapping pair...
        if len(possible_swaps) > 1:
            #remove root not from the list
            if possible_swaps[0] == self.root_node:
                possible_swaps = possible_swaps[1:]
            #select the swapping node (this is the child in the pair)
          
            random_number = np.random.randint(len(possible_swaps))
            swap_node = possible_swaps[random_number]
                       
            #swap the split feature/values and reset indicies
            self.implement_swap(swap_node)
            
            #check it is a valid swap
            for i in self.terminal_nodes:
                #if it is invalid - reverse the swap
                if i.data_indices.size <= min_split:
                    self.implement_swap(swap_node)
                    break
                #if it is valid - set the acceptance              
                else:
                    self.qp_acceptance = 1
            

    def implement_swap(self, swap_node):
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
        
        #reset the data indicies
        self.reset_indices(swap_parent, swap_parent.leftChild, swap_parent.rightChild)
        
              
# ******** RESULTS METHODS *******************      
    
    def train_auc(self):

        y_predicted = np.zeros(self.Y.shape[0])
        
        for i in self.terminal_nodes:

            # get y_predicted for AUC
            if i.class_label == 1:
                np.put(y_predicted, i.data_indices, 1)
        

        auc = metrics.roc_auc_score(self.Y, y_predicted)

        return auc
    
        
    def test_auc(self, X_test, y_test):
    
        self.root_node.test_data_indices = np.arange(X_test.shape[0])
       
        parent = self.root_node
        
        if parent.leftChild is None:
            auc = 0.5

            
            
        else:
            
            self.set_test_indices(parent,parent.leftChild, parent.rightChild, X_test, y_test)
            
            y_predicted = np.zeros(y_test.shape[0])
            
            for term_node in self.terminal_nodes:
               
               if term_node.class_label == 1:
                   node_indices = term_node.test_data_indices
                   np.put(y_predicted, node_indices, 1)
                                      
            auc = metrics.roc_auc_score(y_test, y_predicted)
            
                                 
        return auc
        
            
    def split_test_data(self, parent, feature, threshold, X_test, y_test):
        
        all_leftData = np.where(X_test[:,feature] <= threshold)[0]
        leftData = np.intersect1d(all_leftData, parent.test_data_indices)
        
        all_rightData = np.where(X_test[:,feature] > threshold)[0]
        rightData = np.intersect1d(all_rightData, parent.test_data_indices)
        
        return leftData, rightData
        
    def set_test_indices(self, parent, left, right, X_test, y_test):

        leftData,rightData = self.split_test_data(parent,parent.split_feature, parent.split_value, X_test, y_test)
        
        left.test_data_indices = leftData
        right.test_data_indices = rightData

        
        if left.split_value is None and right.split_value is None:
            return
        
        elif left.split_value is not None and right.split_value is None:
            return self.set_test_indices(left, left.leftChild, left.rightChild, X_test, y_test)
        
        elif left.split_value is None and right.split_value is not None:
            return self.set_test_indices(right, right.leftChild, right.rightChild, X_test, y_test)
        
        else:
            return self.set_test_indices(left, left.leftChild, left.rightChild, X_test, y_test), self.set_test_indices(right, right.leftChild, right.rightChild, X_test, y_test)
            

# ******** DEBUGGING METHODS ******************* 
            
            
    def check_indices(self, parent):
       
        leftData,rightData = self.split_data(parent,parent.split_feature, parent.split_value)
        
        if parent.leftChild is not None:
            if leftData != parent.leftChild.data_indices:
                print "ERROR in indices**************************"
                print "Node " + str(parent.leftChild.id)
                sys.exit(0)
            else:
                self.check_indicesindices(parent.leftChild)
        
        if parent.rightChild is not None:
            if rightData != parent.rightChild.data_indices:
                print "ERROR in indices**************************"
                print "Node " + str(parent.rightChild.id)
                sys.exit(0)
            else:
                self.check_indices(parent.rightChild)
                
    def printTree(self, filename):
        tree_graph = Digraph('Tree', filename=filename, strict=True, format = 'pdf')
        
        if len(self.terminal_nodes) == 1:
            tree_graph.node("root node", shape='box')
        
        else:
        
            for current_node in self.terminal_nodes: 
                
                if current_node.isLeft:
                    label = "<="
                else:
                    label = ">"
        
                tree_graph.node(str(current_node.id), shape='box')
                tree_graph.edge("N " +str(current_node.parent.id) + "\n"
                                + "F " + str(current_node.parent.split_feature) + "\n"
                                + "V " + str(np.round(current_node.parent.split_value,2)), 
                                str(current_node.id), label = label)
                
                current_node = current_node.parent
                
                while current_node != self.root_node:
                    
                    if current_node.isLeft:
                        label = "<="
                    else:
                        label = ">"
                        
                    tree_graph.edge("N " +str(current_node.parent.id) + "\n"
                                    + "F " + str(current_node.parent.split_feature) + "\n"
                                    + "V " + str(np.round(current_node.parent.split_value,2)), 
                                    "N " +str(current_node.id) + "\n"
                                    + "F " + str(current_node.split_feature) + "\n"
                                    + "V " + str(np.round(current_node.split_value,2)), label = label)
                    
                    current_node = current_node.parent  
                
        tree_graph.render()


#****************************************************************************************************
# ACCEPTANCE FUNCTION
#****************************************************************************************************

def acceptance(tree, proposal_tree):

    
    acceptance_calc = proposal_tree.qp_acceptance * proposal_tree.likelihood / tree.likelihood
    
    acceptance = min(1, acceptance_calc)
    
    return acceptance         
                
            
            
        




