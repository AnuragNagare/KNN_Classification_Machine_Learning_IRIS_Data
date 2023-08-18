#!/usr/bin/env python
# coding: utf-8

# <h2 style='color:blue' align="center">KNN (K Nearest Neighbors) Classification: Machine Learning Tutorial Using Sklearn</h2>

# ![iris-dataset.png](attachment:iris-dataset.png)

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# In[2]:


iris.feature_names


# In[3]:


iris.target_names


# In[4]:


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()


# In[5]:


df.shape


# In[6]:


df['target'] = iris.target
df.head()


# In[7]:


df[df.target==1].head()


# In[8]:


df[df.target==2].head()


# In[9]:


df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df.head()


# In[10]:


df[45:55]


# In[11]:


df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## sepal length vs sepal width (Setosa vs Versicolor)

# In[13]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='.')


# ##  petal length vs petal width (Setosa vs Versicolor)

# In[14]:


plt.xlabel('petal Length')
plt.ylabel('petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='.')


# ## Train test split

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target


# In[17]:


X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[18]:


len(X_train)


# In[19]:


len(x_test)


# ## Create KNN (K Nearest Neighbour Classifier)

# In[20]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)


# In[21]:


knn.fit(X_train, y_train)


# In[22]:


knn.score(x_test, y_test)


# In[23]:


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[25]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# You can find this project on <a href='https://github.com/Vyas-Rishabh/KNN_Classification_Machine_Learning_IRIS_Data'><b>Github.</B></a>
