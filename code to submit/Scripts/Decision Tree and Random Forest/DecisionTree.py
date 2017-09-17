from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import pandas as pd
import itertools, time


# *********************************
# ******** read in the data *******
# *********************************
train_data = pd.read_csv("/Users/Frankie/Documents/Dissertation/Data/pancreatic/pancreatic_1_train.csv")
y_train = train_data['label'].as_matrix()
X_train = train_data.drop('label', axis=1).as_matrix()

test_data = pd.read_csv("/Users/Frankie/Documents/Dissertation/Data/pancreatic/pancreatic_1_test.csv")
y_test = test_data['label'].as_matrix()
X_test = test_data.drop('label', axis=1).as_matrix()


# *********************************
# ********    parameters    *******
# *********************************
 
 # number of iterations
iterations = 500

# model variables
split_crit = ('gini', 'entropy')
no_of_features = ('auto', 'log2', None)
depth_of_tree = (100,50, None)
sample_splits = (2,10,20)

# get all combinations of parameters 
cross_val_list =  list(itertools.product(split_crit, no_of_features, depth_of_tree,sample_splits))
no_of_models = len(cross_val_list)

# *********************************
# *** set up arrays for results ***
# *********************************
auc_results = np.zeros((no_of_models,iterations))
runtime = np.zeros((no_of_models))


# *********************************
# *** start running the models  ***
# *********************************
for model_no ,x in enumerate(cross_val_list):

	start = time.time()

	dt = DecisionTreeClassifier(criterion = x[0],
								max_features = x[1],
								max_depth = x[2],
								min_samples_split = x[3] )
	
	# run each model 500 times and record the AUC
	for i in range(0,iterations):
	    dt_model = dt.fit(X_train,y_train)
	    y_pred = dt_model.predict(X_test)
	    auc_results[model_no][i] = metrics.roc_auc_score(y_test, y_pred)

	stop = time.time()

	runtime[model_no] = stop-start


# *********************************
# ******    print results    ******
# *********************************
cross_val_list_np = np.asarray(cross_val_list)
mean = np.mean(auc_results, axis = 1)
stdv = np.std(auc_results, axis = 1)

final_results = np.stack(( runtime, mean, stdv), axis=-1)
final_results = np.concatenate((cross_val_list_np,final_results), axis = 1)
np.savetxt("final_results.txt", final_results, delimiter=',', fmt="%s")