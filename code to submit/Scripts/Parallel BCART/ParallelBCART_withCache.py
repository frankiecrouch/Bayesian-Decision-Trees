from __future__ import division
from BCART_cache import *
from pyspark import SparkContext, SparkConf
import numpy as np
import pandas as pd
import copy, time, sys, os
from functools import partial


time_1 = time.time()
# ****************************************************************************************
# resample function
# ****************************************************************************************

def norm_weights(row):
    weight = row[2]
    row = (row[0], row[1], weight / total_weight_bc.value)
    
    return row

def resample(idx, iter_rows):
    for id, row in enumerate(iter_rows):
        
        tree_id = no_in_part.value[idx] + id
        number = resample_list.value[tree_id] 

        for _ in range(number):
            tree = copy.deepcopy(row[1])
            row = (row[0],tree, 1)
            yield (row)
            
def partition_count(iter_rows):
    total = 0
    for row in iter_rows:
        total += 1

    yield (total)

# ****************************************************************************************

# get parameters
no_of_trees = int(sys.argv[1])
iterations = int(sys.argv[2])
min_split = int(sys.argv[3])
beta = float(sys.argv[4])

# get number of threads
no_threads = sys.argv[5]
run_type = "local[" + run_type + "]"

# get name of folder to save results
save_file = sys.argv[8]

# ****************************************************************************************
# Spark set up and filepaths
# ****************************************************************************************

# location the results will be saved and create folder
save_path = "/Users/Frankie/Documents/Dissertation/Results/BCART_parallel/"
os.mkdir(save_path+save_file)

# filepath of the training and testing data
filepath = "/Users/Frankie/Documents/Dissertation/Data/pancreatic/pancreatic_1_train.csv"
filepath2 = "/Users/Frankie/Documents/Dissertation/Data/pancreatic/pancreatic_1_test.csv"

# spark config
# *** to run in cluster mode: .setMaster("yarn")  ***** 
conf = SparkConf().setMaster(run_type).setAppName("BCART").set("spark.executor.heartbeatInterval","80s")
sc = SparkContext(conf=conf)

# set checkpoint directory
check_dir = sc.setCheckpointDir('checkpoint/')


# ****************************************************************************************
# array for runtime results
# ****************************************************************************************
runtime = np.zeros((7,iterations))


time_2 = time.time()
# ****************************************************************************************
# read in the data
# ****************************************************************************************
data = pd.read_csv(filepath)
features_train = data.drop('label', axis=1).as_matrix()
label_train = data['label'].as_matrix()

data = pd.read_csv(filepath2)
features_test = data.drop('label', axis=1).as_matrix()
label_test = data['label'].as_matrix()


time_3 = time.time()
# ****************************************************************************************
# broadcast variables
# ****************************************************************************************
X_train = sc.broadcast(features_train)
y_train = sc.broadcast(label_train)

X_test = sc.broadcast(features_test)
y_test = sc.broadcast(label_test)

min_split = sc.broadcast(min_split)


time_4 = time.time()
# ****************************************************************************************
# create tree with just root node
# ****************************************************************************************
starting_indices = np.arange(features_train.shape[0])
test_indices = np.arange(features_test.shape[0])
rootNode = Node(data_indices = starting_indices, test_indices = test_indices)
tree = Tree(root_node=rootNode, alpha = 0.95, beta = beta)


# ****************************************************************************************
# copy tree to make the forest
# ****************************************************************************************
forest = []
for i in range (0,no_of_trees):
    tree = copy.deepcopy(tree)
    forest.append((i , tree, 0))
# create rdd of the forest
forest_RDD = sc.parallelize(forest)

time_5 = time.time()

# ****************************************************************************************
# generate starting sample
# collect() to get runtime and the weights to be used in norm step
# ****************************************************************************************
forest_RDD = forest_RDD.map(partial(build_tree, 
                                    X_train = X_train, 
                                    y_train = y_train, 
                                    min_split = min_split)).cache()

forest = forest_RDD.collect()


time_7 = time.time()

# ****************************************************************************************
# get the evaluation metrics 
# ****************************************************************************************
auc_results = np.asarray(forest_RDD.map(partial(calc_auc,
                                    y_train = y_train, 
                                    X_test = X_test,
                                    y_test = y_test)).collect())
auc_test_results = auc_results[:,0]
auc_train_results = auc_results[:,1]

time_8 = time.time()
# ****************************************************************************************
# start iterations
# ****************************************************************************************
for i in range(0,iterations):
    # uncomment to see progress as program runs
    # print "************************************************************************************************"
    # print "iteration " + str(i)
    
    # start timer
    time_9 = time.time()

    # ****************************************************************************************
    # get an array of all the weights and sum and broadcast for normalisation
    # ****************************************************************************************
    
    weights = np.asarray(forest)[:,2]
    total_weights = np.sum(weights)

    total_weight_bc = sc.broadcast(total_weights)

    
    # ****************************************************************************************
    # check if we need to resample
    # ****************************************************************************************
    normalised_weights = np.divide(weights,total_weights)
    n_eff = 1/np.sum(np.square(normalised_weights))

    time_10 = time.time()

    if n_eff < no_of_trees/2:
        # ****************************************************************************************
        # get stratified sample  
        # ****************************************************************************************     
        cdf = np.cumsum(normalised_weights)
        
        start_point = np.random.uniform(0,1/no_of_trees)
        
        l = 0
        resample_list = []
        for j in range(0,no_of_trees):
            sample = start_point + (j)/no_of_trees
            while sample > cdf[l]:
                l += 1
            resample_list.append(l)

        resample_list = np.bincount(resample_list + [no_of_trees-1])
        resample_list[no_of_trees-1] -= 1

        #  broadcast sample list
        resample_list = sc.broadcast(resample_list)

        time_11 = time.time()
        
        # get the number of rows in each partition and broadcast out
        part_number = [0] + list(np.cumsum(np.asarray(forest_RDD.mapPartitions(partition_count).collect())))
        no_in_part = sc.broadcast(part_number)        

        # generate new sample across the partitions
        forest_RDD = forest_RDD.mapPartitionsWithIndex(resample).cache()

        # call action 
        forest_RDD.first()
      
        time_12 = time.time()

        runtime[1][i] = time_11 - time_10
        runtime[2][i] = time_12 - time_11
        runtime[3][i] = 0



    else:
    # ****************************************************************************************     
    # if no resampling done >> normalise the weights
    # ****************************************************************************************     
        forest_RDD = forest_RDD.map(norm_weights).cache()

        forest = forest_RDD.first()

        time_13 = time.time()

        runtime[1][i] = 0
        runtime[2][i] = 0
        runtime[3][i] = time_13 - time_10

    time_14 = time.time()

    # ****************************************************************************************     
    # apply proposal step
    # ****************************************************************************************     
    forest_RDD = forest_RDD.map(partial(tree_proposal,
                                            X_train = X_train, 
                                            y_train = y_train, 
                                            min_split = min_split)).cache()
    

    # checkpoint every 200 iterations
    if i % 200 == 0 :
        forest_RDD.checkpoint()

    forest = forest_RDD.collect()

    time_15 = time.time()

    # ****************************************************************************************
    # get the evaluation metrics 
    # ****************************************************************************************
    auc_results = np.asarray(forest_RDD.map(partial(calc_auc,
                                    y_train = y_train, 
                                    X_test = X_test,
                                    y_test = y_test)).collect())

    auc_test_results_iter = auc_results[:,0]
    auc_train_results_iter = auc_results[:,1]

    auc_test_results  = np.column_stack((auc_test_results,auc_test_results_iter))
    auc_train_results = np.column_stack((auc_train_results, auc_train_results_iter))
    
    time_16 = time.time()

    runtime[0][i] = time_10 - time_9
    
    runtime[4][i] = time_15 - time_14
    runtime[5][i] = time_16 - time_15
    runtime[6][i] = time_16 - time_9

# ****************************************************************************************
# print results
# ****************************************************************************************

# runtime results
np.savetxt((save_path+save_file+"runtime_raw.txt"), runtime, delimiter=',')

time_total = np.sum(runtime, axis = 1)
np.savetxt((save_path+save_file+"runtime_total.txt"), time_total, delimiter=',')

runtime[runtime == 0] = np.nan
means = np.nanmean(runtime, axis=1)
np.savetxt((save_path+save_file+"runtime_average.txt"), means, delimiter=',')

# runtime summary
with open((save_path+save_file+'runtime_summary.txt'), 'w') as f:
    f.write(str(time_16 - time_1) +", Total")
    f.write("\n")
    f.write(str(time_8 - time_1) +", Initialisation" )
    f.write("\n")
    f.write(str(time_16-time_8) +", Total iterations")
    f.write("\n")
    f.write(str(time_2-time_1) +", Set up")
    f.write("\n")
    f.write(str(time_3-time_2) +", Read data")
    f.write("\n")
    f.write(str(time_4-time_3) +", BC data")
    f.write("\n")
    f.write(str(time_5-time_4) +", Build forest")
    f.write("\n")
    f.write(str(time_7-time_5) +", generate sample")
    f.write("\n")
    f.write(str(time_8-time_7) +", auc")
    f.write("\n")
    f.write(str(means[0]) +", norm calc")
    f.write("\n")
    f.write(str(means[1]) +", strat resample")
    f.write("\n")
    f.write(str(means[2]) +", map resample")
    f.write("\n")
    f.write(str(means[3]) +", norm_map")
    f.write("\n")
    f.write(str(means[4]) +", proposal and checkpoint")
    f.write("\n")
    f.write(str(means[5]) +", auc result")
    f.write("\n")
    f.write(str(means[6]) +", total iteration")

# AUC results
np.savetxt((save_path+save_file+"auc.txt"), auc_test_results, delimiter=',')
np.savetxt((save_path+save_file+"auc_train.txt"), auc_train_results, delimiter=',')

max_auc = np.argmax(auc_train_results, axis = 1)

all_results = []
for i in range(0,no_of_trees):
    all_results.append(auc_test_results[i][max_auc[i]])

mean_result = np.average(np.asarray(all_results))
std = np.std(np.asarray(all_results))

# AUC summary
with open((save_path+save_file+'results_summary.txt'), 'w') as f:
    f.write(str(beta) + "," +
            str(min_split.value) + ","+
            str(no_of_trees) + ","+
            str(iterations) + ","+
            str(data_type) + "," +
            str(data_split) + ","+
            str(run_type) + ","+
            str(mean_result) + ","+
            str(std))


# uncomment to print out the first 10 trees
# trees = np.asarray(forest)[:,1]
# trees = trees[:10]
# for k, tree in enumerate(trees):
#     filename = "tree_" + str(i) + "_" + str(k)
#     tree.printTree(filename = filename)
