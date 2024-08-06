#!/usr/bin/env python
# coding: utf-8

# # Task 2: Data_Processing

# In[1]:


# import the Libraries needed
import pandas as pd
import numpy as np


# In[2]:


# Load the Dataset
df = pd.read_csv("../Data/BostonHousing.csv")


# The 'pd.read_csv()' function in pandas is used to read a comma-separated values(CSV) file into a DataFrame.

# # Handline missing values in Dataset
# ## Missing Value
# Missing data/value is defined as the values or data that is not stored (or not present) for some variable/s in the given dataset.
# Missing values in a dataset can be represented in various ways, depending on the soure of the data and conventions used as:
# - NaN (Not a Number)
# - Null or None
# - Empty Strings
# - Blanks or Spaces
# 
# Understanding the representation of missing values in your dataset is crucial for proper data cleaning and preprocessing. Identifying and handling these missing values accurately ensures that your data analysis and machine learning models perform optimally.
# ### Methods for Identifying Missing data
# .isnull() and .notnull() returns a boolean Series of DataFrame.
# 
# - .isnull(): Identifies missing values in a pandas Series or DataFrame \
# (Where 'True' indicates missing values and 'False' indicates non-missing values.)
#              
# - .notnull(): Check for missing values in a pandas Series or DataFrame. \
# (Where 'True' incdicates non-missing values and 'False' indicates missing values.)
# 
# Summerize Missing Values: To get a summary of the number of missing values per column, we can use '.sum()' as: \
# df.isnull().sum()

# In[7]:


df.isnull() # returns boolean Series of DataFrame, where 'True' indicates missing values.


# In[8]:


df.isnull().sum() # Summerize missing values 


# Since all the values are '0', it confirms that there are no missing values in any of the features in dataset.

# In[9]:


df.notnull() # returns boolean Series of DataFrame, where 'True' indicates non-missing values.


# In[10]:


df.notnull().sum() # Summerize non-missing values


# Since all the values are '506' (i.e., equal to total number of rows). It confirms that dataset contains '506' records (non-null values), thus no missing values in any of the features.

# ### Handle the missing values
# #### Drop Rows and Columns with Missing Values

# In[11]:


# Drop rows with any missing values
df_dropped_rows = df.dropna()

print("\nDataFrame after removing rows with missing values:")
df_dropped_rows


# Since there are no missing values in the DataFrame, the 'dropna()' operation does not remove any rows. Therefore, the resulting DataFrame remains unchanged with [506 rows x 14 columns]

# In[12]:


# Drop columns with any missing values
df_dropped_columns = df.dropna(axis=1)

print("\nDataFrame after removing columns with missing values:")
df_dropped_columns


# Since there are no missing values in the DataFrame, the 'dropna(axis=1)' operation does not remove any columns. Therefore, the resulting DataFrame remains unchanged with [506 rows x 14 columns]

# #### Imputation Methods
# - Replacing missing values with estimated values.
# - Preserves sample size: Doesn't reduce data points.
# - Can introduce bias: Estimated values might not be accurate.
# ##### Some Common Imputation methods:
# 1. Mean Imputation: Calculate the mean of required column in the DataFrame. \
# Fills missing values in the column with the mean value.
#     - mean_imputation = df['column_name'].fillna(df['column_name'].mean())
# 2. Median Imputation: Calculates the median of the required column in the DataFrame. \
# Fills missing values in the column with the median value.
#     - median_implutation = df['column_name'].fillna(df['column_name'].median())
# 3. Mode Imputation: Calculate the mode of the required column in the DataFrame. \
# Fills missing values in the column with the mode value.
#     - mode_imputatation = df['column_name'].fillna(df['column_name'].mode())

# # Handling the Outliers
# ## Outliers
# An outlier is a data point that significanlty deviates from the rest of the data. It can be either much higher or much lower than the other data points.
# Or, an outlier is a data point that stands out a lot from the other data points in a set. 
# ### Methods for Identifying Outliers
# #### Visualization
# - **Box plots**: Show the distribution of data and highlight outliers as points outside the whiskers.
# - **KDE plot (Kernel Density Esimate)**:  Provides a smooth, continuous estimate of the data's distribution; outliers may appear in low-density regions or unusual peaks.
# - **Histogram**: Displays the frequency distribution of data, where outliers are identified as significantly smaller or higher bars that are separated from the bulk of the data, often found on the far left or right of the plot.

# In[13]:


# import the Libraries needed
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


# box plot
# Create subplots
fig, axs = plt.subplots(ncols=3, nrows=5, figsize=(20, 20))
index = 0
axs = axs.flatten() # Flattern the 2D array of axes to 1D

# Plot a box plot for each column
for k, col in df.items():
    sns.boxplot(x=k, data=df, ax=axs[index])
    index += 1

# Hide any unused subplots if there are more subplots than columns
for j in range(len(df.columns), len(axs)):
    axs[j].axis('off')  

# Adjust layout
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()


# As we know, Box plot shows the distribution of data and highlight outliers as points outside the whiskers.
# Columns like 'crim', 'zn', 'chas', 'rm', 'dis', 'ptratio', 'b', 'lstat', 'medv' seems to have outliers.

# Another interesing fact on the dataset is the max value of MEDV. From the original data description, it says: Variable #14 seems to be censored at 50.00 (corresponding to a median price of $50,000). Based on that, values above 50.00 may not help to predict MEDV. \
# So, remove 'medv' outliers (medv =50.0) before plotiing more distribuitons.

# In[15]:


# Box plot of 'medv' column
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['medv'])


# In[16]:


# Drop data points that are greater than 50.0
df = df[~(df['medv'] >= 50.0)]
print(np.shape(df))


# In[17]:


# Create subplots
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten() # Flattern the 2D array of axes to 1D

# Plot a kde plot for each column
for k, col in df.items():
    sns.kdeplot(col, ax=axs[index])
    index += 1
#Adjust layout
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()


# In[18]:


# Create subplots
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten() # Flattern the 2D array of axes to 1D

# Plot a kde plot for each column
for k, col in df.items():
    sns.histplot(col, ax=axs[index])
    index += 1
#Adjust layout
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()


# ### Statistical Methods
# - **Z-score**: This method calculates the standard deviation of the data points and identifies outliers as those with Z-Scores exceeding a cetrain threshold (typically 3 or -3). \
# Formula for Z score = (Observation - Mean)/Standard Deviation \
# $z= \frac{X-\mu}{\sigma}$ \
# where: \
# X is the data point \
# $\mu$ is the mean of the data \
# $\sigma$ is the standard deviation of the data
# 
# - **Interquartile Range (IQR)**: IQR identifies outliers as data points falling outside the range of the lower whisker and upper whisker defined by: \
# IQR = $Q3 - Q1$ \
# lower_whisker = $Q1 - {k \times IQR}$ and \
# upper_whisker = $Q3 + {k \times IQR}$ , \
# where Q1 and Q3 are the first and third quartiles, and k is a factor (typically 1.5).

# ##### Calculating Z-Scores

# In[19]:


from scipy.stats import zscore

# Calculate Z-Scores for the 'crim' column
df['crim_zscore'] = zscore(df['crim'])

# Define the thresold for identifying outliers
threshold = 3

# Identify outliers
outliers = df[df['crim_zscore'].abs() > threshold]

# Print the outliers
outliers


# This shows that data points at rows [380, 398, 404, 405, 410, 414, 418, 427] of the 'crim' column are outliers because their Z-scores are significantly high or low compared to the rest of the data or beyond the threshold of $\pm 3$ which can be seen in 'crim_zscore' column at last.

# ##### Calculating IQR

# In[20]:


# Calculate quartiles for 'ptratio' column
Q1 = df['ptratio'].quantile(0.25)
Q3 = df['ptratio'].quantile(0.75)

# Compute IQR
IQR = Q3 - Q1

# Define the thresold for identifying outliers
k = 1.5

# Define bounds for outliers
lower_whisker = Q1 - k * IQR
upper_whisker = Q3 + k * IQR

outliers = df[(df['ptratio'] < lower_whisker) | (df['ptratio'] > upper_whisker)]
print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Lower Whisker: {lower_whisker}")
print(f"Upper Whisker: {upper_whisker}")
outliers


# This shows that data points at rows [196, 197, 198, 258, 259, 260, 261, 262, 263, 264, 265, 266, 268] of the 'ptratio' column are outliers because they are below the Lower Whisker or above the Upper Whisher.

# ### Handle the Outliers
# Outliers, data points that significanlty deviate form the majority, can have detrimental effects on machine learning models. To address this, several techniques can be employed to handle outliers effectively:
# 1. **Removal**
# This involves identifyng and removing outliers from the dataset before training the model.
#     - Thresholding: Outliers are identified as data points exceeding a certain threshold (e.g., Z-score >3).
# 2. **Replacing**
# This involves identifying and replacing outliers with more accurate values.
#     - Replacing with Median: Use to reduce the influence of outliers by replacing them with a central value. \
#     : df.loc[calc_z > thresold, 'column_name'] = df['column_name'].median()
#     - Replacing with Mean: Use when the outliers are not extreme and the mean is a representive value. \
#     : df.loc[calc_z > thresold, 'column_name'] = df['column_name'].mean()
#     - Replacing with a Threshold: Replce outliers with a predefined threshold value. \
#     : df.loc[df['column_name'] > threshold_value, 'column_name'] = threshold_value
# 
# 2. **Transformation**
# This involves transforming the data to reduce the influence of outliers.
#     - Log transformation: Applying a logarithmic transformation to compress the data and reduce the impact of extreme values. 
#         - Suitable for: Positive skewness.
#         - Formula: Transformed Value = log(x + constant)
#     - Square Root Transformation:
#         - Suitable for: Moderated positive skewness.
#         - Formula: Transformed Value = $\sqrt{x + constant}$
#     - Cube Root Transformation:
#         - Suitbale for: Both positive and negative skewness.
#         - Formula: Transformed Value = $\sqrt[3]{x + constant}$
#     - Yeo-Jhonson Transformation:
#         - Suitable for both positve and negative skewness and handle zero/negative values.
#     
#     - Scaling: Standardizing or normalizing the data to have a mean of zero and a standard deviation of one.
# 

# ##### Removing Outliers Using Z-Score
# - Compute Z-scores for the column
# - Filter the data to exclude rows where the Z-score exceeds the threshold

# In[21]:


from scipy.stats import zscore

# Calculate Z-Scores for the 'crim' column
df['crim_zscore'] = zscore(df['crim'])

# Define the thresold for identifying outliers
threshold = 3

# Remove outliers
df_filtered = df[df['crim_zscore'].abs() <= threshold]

# Drop the Z-score column if no loger needed
df_filtered = df_filtered.drop(columns=['crim_zscore'])

# Print results
print("Filtered Data (Outliers Removed):")
df_filtered


# Since data points at rows [380, 398, 404, 405, 410, 414, 418, 427] of the 'crim' column are outliers, total number of Outliers is 8. \
# df_filtered display the dataset after removing the rows that are outliers.\
# thus, \
# Filtered_rows = Total no of rows - total no of outliers \
# Filtered_rows = 490 - 8 \
# Filtered rows = 482

# #### Another way to remove outliers using Z-score

# In[22]:


# Calculate Z-Scores for the 'crim' column
df['crim_zscore'] = zscore(df['crim'])

# Define the thresold for identifying outliers
threshold = 3

# Identify outliers
outliers = df[df['crim_zscore'].abs() > threshold]

#drop rows containing outliers
df_filtered = df.drop(outliers.index)

# Drop the 'crim_zscore' column
df_filtered = df_filtered.drop(columns=['crim_zscore'])

# Print results
print("Filtered Data (Outliers Removed):")
df_filtered


# ##### Removing Outliers Using IQR
# - Compute the IQR and determine lower and upper whishkers.
# - Filter the data to exclude rows outside these whishkers (bounds).

# In[23]:


# Calculate quartiles for 'ptratio' column
Q1 = df['ptratio'].quantile(0.25)
Q3 = df['ptratio'].quantile(0.75)

# Compute IQR
IQR = Q3 - Q1

# Define the thresold for identifying outliers
k = 1.5

# Define bounds for outliers
lower_whisker = Q1 - k * IQR
upper_whisker = Q3 + k * IQR

# Remove outliers
df_filtered = df[(df['ptratio'] >= lower_whisker) & (df['ptratio'] <= upper_whisker)]

# Print results
print("Filtered Data (Outliers Removed):")
df_filtered


# Since data points at rows [196, 197, 198, 258, 259, 260, 261, 262, 263, 264, 265, 266, 268] of the 'ptratio' column are outliers, total number of Outliers is 15.
# df_filtered display the dataset after removing the rows that are outliers.\
# thus, \
# Filtered_rows = Total no of rows - total no of outliers \
# Filtered_rows = 490 - 13 \
# Filtered rows = 477

# #### Another way to remove outliers using IQR

# In[24]:


# Calculate quartiles for 'ptratio' column
Q1 = df['ptratio'].quantile(0.25)
Q3 = df['ptratio'].quantile(0.75)

# Compute IQR
IQR = Q3 - Q1

# Define the thresold for identifying outliers
k = 1.5

# Define bounds for outliers
lower_whisker = Q1 - k * IQR
upper_whisker = Q3 + k * IQR

outliers = df[(df['ptratio'] < lower_whisker) | (df['ptratio'] > upper_whisker)]

# drop rows containing outliers
df_filtered = df.drop(outliers.index)

# Print results
print("Filtered Data (Outliers Removed):")
df_filtered


# In[25]:


# Drop the 'crim_zscore' column from the DataFrame
df = df.drop(columns=['crim_zscore'])


# ### Transformation
# **Skewness** is a measure of the asymmetry of the distribution of data around its mean.
# - Positive Skewness: The distribution has a longer or fatter tail on the right side.
# - Negative Skewness: The distribution has a longer or fatter tail on the left side.
# - Zero Skewness: The distribution is symmetrical.
# 
# **Step 1: Identify Skewness Categories**
# - Highly Positive Skewness: Skewness greater than 1.0
# - Moderate Positive Skewness: Skewness between 0.5 and 1.0
# - Highly Negative Skewness: Skewness less than -1.0
# - Moderate Negative Skewness: Skewness between -1.0 and -0.5

# In[26]:


# Heatmap that displays the correlation between target 'medv' and other features
plt.figure(figsize=(10,10))
sns.heatmap(df.corr().abs(), annot = True)


# Reduce or remove the skewness for only those columns where the correlation with the target is not that great.

# In[27]:


# calculates the skewness of each column in a DataFrame.
df.skew()


# From above data of skewness, it is clear that:
# - Hignly Positve Skew = ['crim', 'zn', 'chas']
#     - Here, 'chas' contains binary data '0' and '1' only so no transfomation is needed.
# - Moderated Positive Skew = ['indus', 'nox', 'rm', 'dis' 'rad', 'tax', 'lstat', 'medv']
# - Moderated Negative Skew = ['age', 'ptratio']
# - Higly Negative Skew = ['b']

# **Step 2: Apply Transformations**

# In[28]:


import numpy as np
import pandas as pd
from scipy.stats import mstats
from scipy.stats import boxcox


# Create a copy of the original DataFrame
df_transformed = df.copy()

# Define columns with different types of skewness
highly_positive_skew = ['crim','zn']
moderate_positive_skew = ['indus', 'nox', 'dis', 'rad', 'tax', 'lstat']
moderate_negative_skew = ['age', 'ptratio']
highly_negative_skew = ['b']

# Apply Transformation to the copied DataFrame
#df['crim'] = np.log(df['crim'])
#df['medv'] = np.log(df['medv'])

# For Highly Positive Skew
for column in highly_positive_skew:
    df[column] = np.log(df[column] + 1e-9)

# For Moderate Positive Skew
for column in moderate_positive_skew:
    df[column] = np.sqrt(df[column] - df[column].min() + 1)

# Apply Winsorization
df['rm'] = mstats.winsorize(df['rm'], limits=[0.05, 0.05])
df['medv'] = mstats.winsorize(df['medv'], limits=[0.05, 0.05])

# For Moderate Negative Skew
for column in moderate_negative_skew:
    df[column] = np.sqrt(df[column].max() - df[column] + 1)

# For Highly Negative Skew
for column in highly_negative_skew:
    df[column] = np.log1p(df[column].max() - df[column] + 1)

# df['medv'], lambda_ = boxcox(df['medv'] + 1e-9)

# Display the DataFrame with transformed columns
print(df.head())


# In[29]:


# box plot
# Create subplots
fig, axs = plt.subplots(ncols=3, nrows=5, figsize=(20, 20))
index = 0
axs = axs.flatten() # Flattern the 2D array of axes to 1D

# Plot a box plot for each column
for k, col in df.items():
    sns.boxplot(x=k, data=df, ax=axs[index])
    index += 1

# Hide any unused subplots if there are more subplots than columns
for j in range(len(df.columns), len(axs)):
    axs[j].axis('off')  

# Adjust layout
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()


# ### Outlier detection  is crucial in machine learning for the following reasons:
# - **Biased Models**: Outliers can skew the model towards extreme values, affecting overall performace.
# - **Reduce Accuracy**: They introduce noise, making it hard for the model to learn accurate patterns.
# - **Increase Variance**: Outliers can increase model variance, leading to instability and overfitting.
# - **Reduced Interpretability**: They complicate understanding the model's behavious and predictions. 

# # Encode Categorical Variables
# 1. Label Encoding: Converts categories to numerical labels. Useful for ordinal variables with a meaningful order.
# 
# 2. One-Hot Encoding: Creates binary columns for each category. Ideal for nominal variables with no order.
# 
# 3. Binary Encoding: Converts categories to binary codes, suitable for high-cardinality variables.
# 
# 4. Ordinal Encoding: Maps categories to integers based on their order. Useful for ordinal variables with a meaningful sequence.

# # Normalize/Standardize numerical features.
# **Normalization**
# - Rescaling features to a specific range, typically [0, 1] or [-1, 1].
# - Method: Min-Max Scaling
# - Formula: $X_{\text{norm}} = \frac{X-X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$
# - Use Case: Useful when features have different units or ranges; often used with algorithms sensitive to feature scaling (e.g., neural networks).
# 
# **Standardization**
# - Rescaling features to have a mean of 0 and a standard deviation of 1.
# - Method: Z-score Standardization
# - Formula: $X_{\text{std}} = \frac{X - \mu}{\sigma}$
# - Use case: Useful when features have different means and variances; often used with algorithms assuming normally distributed data (e.g., linear regression)

# In[30]:


df.skew()


# In[31]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
#Columns to transform
normalize_column = ['zn', 'nox', 'dis', 'rad', 'ptratio', 'b', 'medv']
standardize_column = ['crim', 'rm', 'age', 'tax', 'lstat']

# Apply Normalization
N_scaler = MinMaxScaler()
df[normalize_column] = N_scaler.fit_transform(df[normalize_column])

# Apply Standardization
St_scaler = StandardScaler()
df[standardize_column] = St_scaler.fit_transform(df[standardize_column])

print(df)


# In[32]:


# Writing to a csv file
df.to_csv('../data/BostonHousingClean.csv', index=False)


# In[33]:


from sklearn import preprocessing
# Let's scale the columns before plotting them against MEDV
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['lstat', 'indus', 'nox', 'ptratio', 'rm', 'tax', 'dis', 'age']
x = df.loc[:,column_sels]
y = df['medv']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# # Split the data into training and testing sets.
# **Dataset Splitting**
# Scikit-learn alias sklearn is the most useful and robust library for machine learning in Python. The scikit-learn library provides us with the model_selection module in which we have the splitter function train_test_split().

# In[34]:


# Import Libraries
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[35]:


# get the features and target
X = df.drop(columns='medv') # Features
y = df['medv'] # Target Variable


# Here df.drop(column='medv') removes the 'medv' column form the DataFrame 'df'. The resulting DataFrame 'X' contains all columns except 'medv', which are used as features (predictors) for the model. \
# The Series y = df['medv'] is the target variable that the model aims to predict, where df['medv'] selects the 'medv' column form the DataFrame 'df'.

# In[36]:


# Split my data into the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# This line of code splits the data into training and testing sets. \
# This split allows you to train your model on X_train and y_train, and then evaluate its performance on the separate test set X_test and y_test. Here,
# - 'train_test_split(X, y, test_size=0.2, random_state=42)':
#     - X: Features (predictors).
#     - y: Target variable.
#     - test_size=0.2: Specifies that 20% of the data should be used for testing, while the remaining 80% is used for training.
#     - random_state=42: Ensures the split is reproducible by setting a seed for random number generation.
# 
# The function returns four outputs:
# - X_train: Features for training.
# - X_test: Features for testing.
# - y_train: Target values for training.
# - y_test: Target values for testing.
