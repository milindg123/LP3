#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Implement K-Means clustering/ hierarchical clustering on sales_data_sample.csv dataset. 
# Determine the number of clusters using the elbow method.
# Dataset link : https://www.kaggle.com/datasets/kyanyoga/sample-sales-data


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from yellowbrick.cluster import KElbowVisualizer


# In[19]:


data = pd.read_csv('sales_data_sample.csv', sep = ',', encoding = 'Latin-1')
data


# In[20]:


# Prepare the data as needed (feature selection, preprocessing, etc.)


# In[21]:


# Step 2: Select relevant features for clustering (e.g., 'QUANTITYORDERED', 'PRICEEACH')
selected_features = data[['QUANTITYORDERED', 'PRICEEACH']]
selected_features


# In[22]:


# Step 3: Normalize the data (if needed)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_features = scaler.fit_transform(selected_features)


# In[23]:


normalized_features


# In[24]:


# Step 4: Determine the optimal number of clusters using the elbow method
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(normalized_features)
    wcss.append(kmeans.inertia_)


# In[25]:


# Plot the elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[26]:


# Step 5: Choose the optimal number of clusters (elbow point) and perform K-Means clustering
optimal_clusters = 3  # Adjust based on the elbow point in the graph
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(normalized_features)


# In[27]:


# Step 6: Visualize the clusters (if possible)
plt.scatter(normalized_features[:, 0], normalized_features[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('QUANTITYORDERED')
plt.ylabel('PRICEEACH')
plt.title('K-Means Clustering')
plt.show()

