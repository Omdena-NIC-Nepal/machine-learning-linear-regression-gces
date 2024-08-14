#!/usr/bin/env python
# coding: utf-8

# # Task 4: Model Evaluation
# Steps:
# - Evaluate the model using metrics such as Mean Squared Error (MSE), R-squared.
# - Plot residuals to check the assumptions of linear regression.
# - Compare model performance with different feature sets or preprocessing steps.
# - Script: scripts/evaluate_model.py

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[15]:


boston = pd.read_csv('../data/BostonHousing.csv')


# In[16]:


# Get the feature and target
X = boston.drop(columns='medv') # Features
y = boston['medv'] #Target variable


# In[17]:


# Split the data into train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


# Initialize the model
model = LinearRegression()
model.fit(X_train, y_train) # Train the model with the dataset


# In[19]:


# Prediction
y_pred = model.predict(X_test)


# ## To evaluate the model's performance and diagnose potential issues, we'll use metrics such as Mean Squared Error (MSE) and R-squared.

# In[20]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r_square = r2_score(y_test, y_pred)


# In[21]:


# Print the values
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r_square}')


# ## To check the assumptions of linear regression, we'll plot the residuals using a residual plot. We'll use the following code:

# In[22]:


# Import necessary libraries
import matplotlib.pyplot as plt

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# ### Here we use the data that has been preprocessed and cleaned in the previous steps. to compare with the one that hasn't been preprocessed

# In[23]:


df = pd.read_csv('../data/BostonHousingClean.csv')


# In[24]:


# get the features and target
X = df.drop(columns='medv') # Features
y = df['medv'] # Target Variable


# In[25]:


# Split my data into the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[26]:


# train the model
model = LinearRegression() # initiliaze the model
model.fit(X_train, y_train) # Train my model with the dataset for training


# In[27]:


# Prediction
y_pred = model.predict(X_test)


# In[28]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Print the values
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# In[ ]:




