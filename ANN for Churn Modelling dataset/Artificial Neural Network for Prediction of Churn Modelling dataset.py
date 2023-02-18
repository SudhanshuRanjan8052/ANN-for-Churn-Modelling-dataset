#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network For Prediction of whether the people leave the bank or not on the basis of churn modelling dataset

# # Importing Libraries

# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[3]:


tf.__version__


# # Data Preprocessing

# In[4]:


dataset = pd.read_csv("Churn_modelling.csv")


# In[5]:


x= dataset.iloc[:,3:-1].values #independent variable matrix or feature matrix
y= dataset.iloc[:,-1].values #dependent variable matrix or actual outcome matrix


# In[6]:


print(x)


# In[7]:


print(y)


# # Encoding Categorical Data

# Encoding "Gender" Column using Label Enncoding

# In[8]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() #making object of label encoder
x[:,2]=le.fit_transform(x[:,2]) #providing column on which we want to apply it


# In[9]:


print(x)


# Encoding "Country" Column using One Hot Encoding

# In[10]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Encoder",OneHotEncoder(),[1])],remainder="passthrough") #making an object of columntransformer
x = np.array(ct.fit_transform(x)) #appliying fit and transform on desired column


# In[11]:


print(x)


# # Splitting Dataset into Training set and Set set

# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# # Features Scaling

# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# # Building ANN Network

# In[14]:


#Intializing ANN
ann = tf.keras.models.Sequential()


# In[15]:


#Adding input layers and hidden layer
ann.add(tf.keras.layers.Dense(units=7,activation="relu"))


# In[17]:


#Adding another hidden layer
ann.add(tf.keras.layers.Dense(units=7,activation="relu"))


# In[18]:


#Adding output layer
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))


# # Training ANN network

# In[19]:


#compiling the neural network
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


# In[20]:


#training the neural network
ann.fit(x_train,y_train,batch_size=32,epochs=200)


# # Prediction

# In[22]:


#Making the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = ann.predict(x_test)
y_pred = (y_pred>0.5)
cm = confusion_matrix(y_pred,y_test)
print(cm)

#finiding accuracy score
acc = accuracy_score(y_test,y_pred)
print("Accuracy Score: ",acc)


# Use our ANN model to predict if the customer with the following informations will leave the bank: 
# Geography: France
# 
# Credit Score: 600
# 
# Gender: Male
# 
# Age: 40 years old
# 
# Tenure: 3 years
# 
# Balance: $ 60000
# 
# Number of Products: 2
# 
# Does this customer have a credit card? Yes
# 
# Is this customer an Active Member: Yes
# 
# Estimated Salary: $ 50000
# 
# Solution:

# In[23]:


print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)


# It means person will not leave the bank.

# In[ ]:




