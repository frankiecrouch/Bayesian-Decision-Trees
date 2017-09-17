from __future__ import division
import numpy as np
from graphviz import Digraph


class Node(object):

    def __init__(self,isLeft=None,parent=None,split_feature=None,split_value=None, data_indices = None, test_indices = []):

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
        
        self.test_data_indices = test_indices
        self.class_label = None

        self.terminal_id = None


class Tree(object):

    def __init__(self, root_node, alpha, beta):
        self.root_node = root_node
        self.terminal_nodes = [root_node]
        self.internal_nodes = []
        
        self.alpha = alpha
        self.beta = beta

        self.likelihood = 0
        self.qp_acceptance = 1

    def printTree(self, filename):
        tree_graph = Digraph('Tree', filename=filename, strict=True, format = 'pdf')
        
        if len(self.terminal_nodes) == 1:
            tree_graph.node("root node", shape='box')
        
        else:
    
            for idx, current_node in enumerate(self.terminal_nodes):
                
                current_node.terminal_id = idx
                
                if current_node.isLeft:
                    label = "<="
                    color= "green"
                else:
                    label = ">"
                    color = "red"
        
                tree_graph.node(str(current_node.terminal_id), shape='box')
                tree_graph.edge("F " + str(current_node.parent.split_feature) + "\n"
                                + "V " + str(np.round(current_node.parent.split_value,2)), 
                                str(current_node.terminal_id), label = label, color = color)
                
                current_node = current_node.parent
                
                while current_node != self.root_node:
                    
                    if current_node.isLeft:
                        label = "<="
                        color= "green"
                    else:
                        label = ">"
                        color = "red"

                    tree_graph.edge("F " + str(current_node.parent.split_feature) + "\n"
                                    + "V " + str(np.round(current_node.parent.split_value,2)), 
                                    "F " + str(current_node.split_feature) + "\n"
                                    + "V " + str(np.round(current_node.split_value,2)), label = label, color = color)
                    
                    current_node = current_node.parent  
                
        tree_graph.render()
  
                
            
            
        




