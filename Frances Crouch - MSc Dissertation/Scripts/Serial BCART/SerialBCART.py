from BayesianTree import Node, Tree, acceptance
import numpy as np
import pandas as pd
import copy
import time


start_time = time.time()

#****************************************************************************************************
# read in the data 
#****************************************************************************************************
train_data = pd.read_csv("/Users/Frankie/Documents/Dissertation/Data/pancreatic/pancreatic_1_train.csv")
y_train = train_data['label'].as_matrix()
X_train = train_data.drop('label', axis=1).as_matrix()

test_data = pd.read_csv("/Users/Frankie/Documents/Dissertation/Data/pancreatic/pancreatic_1_test.csv")
y_test = test_data['label'].as_matrix()
X_test = test_data.drop('label', axis=1).as_matrix()

end_data_load = time.time()
#****************************************************************************************************
# set parameters: no. of iterations, no. of repeats, alpha and beta
#****************************************************************************************************
iterations = 5000
repeat = 500

alpha = 0.95
beta = 1.5

#****************************************************************************************************
# create arrays to store results: 
# - AUC on the training data
# - AUC on the testing data
# - runtime
#****************************************************************************************************
results_auc_train = np.zeros((repeat , iterations), dtype = np.object)
results_auc_test = np.zeros((repeat , iterations), dtype = np.object)
results_runtime = np.zeros((repeat , iterations), dtype = np.float)


#****************************************************************************************************
# create the starting tree with just a root node
#****************************************************************************************************
starting_indices = np.arange(X_train.shape[0])
rootNode = Node(data_indices = starting_indices)

tree = Tree(root_node=rootNode, alpha = alpha, beta = beta, X=X_train, Y=y_train)
tree.calc_likelihood()


#****************************************************************************************************
# start the iterations
#****************************************************************************************************
for j in range(0,repeat):
    
#   create a copy of the root node tree add the beginning of each MCMC chain   
    current_tree = copy.deepcopy(tree)
    
    for i in range (0,iterations):
 
#       uncomment to see progress as script is running       
        print "repeat " +str(j) + " iteration  " +str(i)
        
#       start timer 
        start = time.time()
        
#       generate the candidate tree
        candidate_tree = copy.deepcopy(current_tree)

#       CHANGE, PRUNE, CHANGE OR SWAP
        random_proposal = np.random.randint(4)
               
        if random_proposal == 0:
            
            candidate_tree.grow()
            
        elif random_proposal == 1:
            
            candidate_tree.prune()
            
        elif random_proposal == 2:
            
            candidate_tree.change()
            
        elif random_proposal == 3:
            
            candidate_tree.swap()
        
        # update the likelihood of the candidate tree
        candidate_tree.calc_likelihood()
        
        # calc acceptance
        acpt = acceptance(current_tree, candidate_tree)
        
        # generate random number
        random_acceptance = np.random.uniform(0,1)
        
        # update tree if accepting
        if random_acceptance < acpt:
            current_tree = copy.deepcopy(candidate_tree)

        # uncomment to print the tree
        # filename = "tree_" + str(i)
        # current_tree.printTree(filename = filename)
        
        #end timer
        stop = time.time()
        
#        record the results
        auc_train = current_tree.train_auc()
        auc_test = current_tree.test_auc(X_test, y_test)

        results_auc_train[j][i] =  auc_train
        results_auc_test[j][i] =  auc_test
        results_runtime[j][i] = (stop-start)

end_total = time.time()

#****************************************************************************************************
# find the best tree from each chain by chosing the tree with the max AUC
#****************************************************************************************************
arg_max_auc = np.argmax(results_auc_train, axis = 1)

all_results = []
for i in range(0,repeat):
    all_results.append(results_auc_test[i][arg_max_auc[i]])

# calculate the average AUC and stdv
mean_result = np.average(np.asarray(all_results))
std = np.std(np.asarray(all_results))
    

#****************************************************************************************************
# export results  
#****************************************************************************************************

# raw data
np.savetxt("auc_test.txt", results_auc_test, delimiter=',')
np.savetxt("auc_train.txt", results_auc_train, delimiter=',')
np.savetxt("runtime.txt", results_runtime, delimiter=',')


# summary of the runtime results
total_iterations_time = (np.sum(results_runtime))/60
min_chain = (np.min(np.sum(results_runtime, axis=1)))/60
max_chain = (np.max(np.sum(results_runtime, axis=1)))/60
ave_chain = (np.mean(np.sum(results_runtime, axis=1)))/60
total_runtime = (end_total - start_time)/60
load_data_time = (end_data_load - start_time)/60

with open('time_results.txt', 'w' ) as f:
    f.write("Total runtime was %f minutes" % total_runtime)
    f.write(", which is %f hours \n" % (total_runtime/60))
    f.write("The data load took %f minutes \n" % load_data_time)
    f.write("The total time spent doing the MCMC chains was %f minutes \n" % total_iterations_time)
    f.write("The min, max and average MCMC chain of length %d was: %f, %f, %f minutes" % (iterations, min_chain, max_chain,ave_chain))


# summary of the prediction results
with open(('results_summary.txt'), 'w') as f:
    f.write('beta, AUC, stdv \n')    
    f.write(str(beta) + "," +
            str(mean_result) + ","+
            str(std))

