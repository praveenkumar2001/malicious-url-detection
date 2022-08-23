
# coding: utf-8

# In[1]:

import pickle
import pandas as pd
import numpy as np
import sklearn
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv("data.csv")
print (df.shape)
data = df.values


# In[8]:


X1 = data[:,0:5]
X2 = data[:,7:15]
print("uuuu",len(X1[1]))
print("lll",len(X2[1]))

X = np.append(X1,X2,axis=1)
print (X1.shape)
print (X2.shape)
print (X.shape)
scaler = StandardScaler().fit(X)
#scaler = Normalizer().fit(X)
rescaledX = scaler.transform(X)
print (rescaledX[0:5,:])


# In[9]:


X3 = data[:,15]
cipher_lst=["AES128-GCM-SHA256","AES128-SHA","AES128-SHA256","AES256-GCM-SHA384","AES256-SHA","AES256-SHA256","ES-CBC3-SHA","DHE-RSA-AES256-GCM-SHA384","DHE-RSA-AES256-SHA","ECDHE-ECDSA-AES128-GCM-SHA256","ECDHE-ECDSA-AES256-GCM-SHA384","ECDHE-RSA-AES128-GCM-SHA256","ECDHE-RSA-AES128-SHA256","ECDHE-RSA-AES256-GCM-SHA384","ECDHE-RSA-AES256-SHA","ECDHE-RSA-AES256-SHA384"]

temp = [[0]*len(cipher_lst)]*len(X3)
for j in range(0, len(X3)):
	for i in range(0,len(cipher_lst)):
		if X3[j]==cipher_lst[i]:
			temp[j][i]=1
		

# In[10]:

print("\n========= ", len(cipher_lst))
X=np.append(X,temp,axis=1)
print (X.shape)


# In[11]:

k1 = int(len(X)/2)
k2 = int(len(X) - len(X)/2)

Y = data[:,17]
Y = np.array(Y)
print (Y.shape)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42)
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)

#clf2 = MultinomialNB().fit(X_train, y_train) 

clf3=RandomForestClassifier()
clf3.fit(X_train, y_train)
pred1 = clf1.predict(X_test)
#pred2 = clf2.predict(X_test)
pred3 = clf3.predict(X_test)
acc1 = accuracy_score(pred1, y_test)*100
#acc2 = accuracy_score(pred2, y_test)*100
acc3 = accuracy_score(pred3, y_test)*100

print(acc1)
#print(acc2)
print(acc3)
#print(pred)
#print(y_test)

filename = 'dt_model_pa.sav'
pickle.dump(clf1, open(filename, 'wb'))  


filename = 'rf_model_pa.sav'
pickle.dump(clf3, open(filename, 'wb')) 

