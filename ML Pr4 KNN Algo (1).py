#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Implement K-Nearest Neighbors algorithm on diabetes.csv dataset. Compute confusion 
# matrix, accuracy, error rate, precision and recall on the given dataset.
# Dataset link : https://www.kaggle.com/datasets/abdallamahgoub/diabetes


# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# In[10]:


data=pd.read_csv("diabetes.csv")
data


# In[11]:


X = data.drop("Outcome", axis=1)  # Features
y = data["Outcome"]  # Target variable


# In[12]:


X


# In[13]:


# 2. Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


# 3. Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[15]:


X_train


# In[16]:


# 4. Implement K-Nearest Neighbors (KNN)
k = 3  # Choose the number of neighbors (k) based on your needs
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)


# In[17]:


# 5. Predict and Evaluate
y_pred = knn.predict(X_test)
y_pred


# In[18]:


# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate accuracy, error rate, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)


# In[19]:


# Accuracy: This measures the overall correctness of the classifier's predictions. In this case, the model is about 70.78% accurate.

# Error Rate: The error rate is the complement of accuracy (1 - accuracy), representing the proportion of incorrect predictions. Here, the error rate is approximately 29.22%.

# Precision: Precision measures the ratio of true positive predictions to the total number of positive predictions (true positives + false positives). A higher precision indicates that when the model predicts the positive class, it's more likely to be correct. In this case, the precision is approximately 60.87%.

# Recall: Recall measures the ratio of true positive predictions to the total number of actual positive instances (true posit


# In[ ]:




