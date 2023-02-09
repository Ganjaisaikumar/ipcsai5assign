#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("C:\\Users\\sai\\Downloads\\NNDL_Code and Data\\NNDL_Code and Data\\glass.csv")


X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)


score = accuracy_score(y_test, y_pred)
print("Accuracy score:", score)


print("Classification report:")
print(classification_report(y_test, y_pred))


# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


glass_data = pd.read_csv("C:\\Users\\sai\\Downloads\\NNDL_Code and Data\\NNDL_Code and Data\\glass.csv")

x_train = glass_data.drop("Type", axis=1)
y_train = glass_data['Type']
# splitting train and test data using train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# Train the model using the training sets
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
# Classification report 
qual_report = classification_report(y_test, y_pred, zero_division = 0)
print(qual_report)
print("SVM accuracy is: ", accuracy_score(y_test, y_pred))


# In[ ]:


#The choice of the algorithm dependsn on the nature of the data and the problem you are trying to solve. 
#In this case, SVM works well.

