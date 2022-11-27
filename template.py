#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def load_dataset(dataset_path):
	#To-Do: Implement this function
	return pd.read_csv(dataset_path)

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	target = dataset_df['target'].value_counts().sort_index()
	return (len(data.columns)-1, target[0], target[1])

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	y = dataset_df['target']
	x = dataset_df.drop("target", axis=1)
	 
	train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=testset_size)
 
	return (train_x, test_x, train_y, test_y)
	

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	clf = DecisionTreeClassifier(random_state=0)
	clf.fit(x_train, y_train)
 
	acc = accuracy_score(y_test, clf.predict(x_test))
	prec = precision_score(y_test, clf.predict(x_test))
	recall = recall_score(y_test, clf.predict(x_test))
 
	return (acc, prec, recall)

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	clf = RandomForestClassifier(random_state=0)
	clf.fit(x_train, y_train)
 
	acc = accuracy_score(y_test, clf.predict(x_test))
	prec = precision_score(y_test, clf.predict(x_test))
	recall = recall_score(y_test, clf.predict(x_test))
 
	return (acc, prec, recall)
 
def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	clf = make_pipeline(
		StandardScaler(),
		SVC()
	)
	clf.fit(x_train, y_train)

	acc = accuracy_score(y_test, clf.predict(x_test))
	prec = precision_score(y_test, clf.predict(x_test))
	recall = recall_score(y_test, clf.predict(x_test))
 
	return (acc, prec, recall)

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)