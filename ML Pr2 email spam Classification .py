#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Practical Number 2 : Email Spam Classification


# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df = pd.read_csv("./emails.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


X = df.iloc[:,1:3001]
X


# In[6]:


Y = df.iloc[:,-1].values
Y


# In[7]:


train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.25)


# In[8]:


svc = SVC(C=1.0,kernel='rbf',gamma='auto')         
# C here is the regularization parameter. Here, L2 penalty is used(default). It is the inverse of the strength of regularization.
# As C increases, model overfits.
# Kernel here is the radial basis function kernel.
# gamma (only used for rbf kernel) : As gamma increases, model overfits.
svc.fit(train_x,train_y)
y_pred2 = svc.predict(test_x)
print("Accuracy Score for SVC : ", accuracy_score(y_pred2,test_y))


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)


# In[10]:


knn = KNeighborsClassifier(n_neighbors=7)


# In[11]:


knn.fit(X_train, y_train)


# In[12]:


print(knn.predict(X_test))


# In[13]:


print(knn.score(X_test, y_test))


# In[ ]:




