import sys
import csv
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

import numpy as np

if __name__=="__main__":
	X = []
	Y = []
	
	fileName = sys.argv[1]
	
	with open(fileName, 'r') as csvfile: 
		csvreader = csv.reader(csvfile, delimiter='\t') 
		for row in csvreader: 
			properties = row[1:-1]
			label = row[-1]
			print(len(properties), properties[0])
			X.append(properties)
			Y.append([label])
	
	X = np.array(X)
	X = X.astype(float)
	Y = np.array(Y)
	Y = Y.astype(int)
	X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.25, random_state = 95)
	print(X.shape)
	
	
	#DecisionTree Classifier
	DT_clf = DecisionTreeClassifier()
	DT_clf.fit(X_train, y_train)

	pred = DT_clf.predict(X_test)
	accDT = accuracy_score(pred, y_test)*100

	print(accDT)
	
	#RandomForest Classifier
	RF_clf = RandomForestClassifier()
	RF_clf.fit(X_train, y_train)

	pred = RF_clf.predict(X_test)
	accRF = accuracy_score(pred, y_test)*100

	print(accRF)

	#KNN Classifier
	KNN_clf = KNeighborsClassifier()
	KNN_clf.fit(X_train, y_train)

	pred = KNN_clf.predict(X_test)
	accKNN = accuracy_score(pred, y_test)*100

	print(accKNN)	
	
	##--------saving the model------------##

	filename = 'dt_model.sav'
	pickle.dump(DT_clf, open(filename, 'wb'))  


	filename = 'rf_model.sav'
	pickle.dump(RF_clf, open(filename, 'wb')) 

	filename = 'knn_model.sav'
	pickle.dump(KNN_clf, open(filename, 'wb'))  
	
	print("\n=====================")
	
	with open('trainingResult.txt', 'w') as f:
		f.write('resulst of trained model are as follows...')
		f.write('DT model : '+str(accDT))
		f.write('RF model : '+str(accRF))
		f.write('KNN model : '+str(accKNN))
	print("done")
