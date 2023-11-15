#!/usr/bin/env python
# coding: utf-8

# 
# House Price Prediction - Advanced regression Assignment
# ------------------------------------------------------------------------------------------------------
# Overview
# ---------
# A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. The company is looking at prospective properties to buy to enter the market. 
# Objective
# ----------
# This research aims to build a regression model using regularisation in order to predict the actual value of the prospective properties and decide whether to invest in them or not.
# 
# Linear regression steps followed
# 
#     1.Data Overview
#         Understanding and loading of Data
# 
#     2.Data Preperation
#         a) Missing Value check
#         b) Outlier detection
# 
#     3.Data Visualization
#         
#     4.Data Encoding
#         dummy variable
# 
#     5.Preparing the model
#         Train-test split
#         rescalling
# 
#     6.Building the model
#         Recursive Feature Estimation
#     
#     7.Ridge and Lasso Regression
#         
#      Ridge
#     
#      Lasso
# 
#     8. Model Evaluation
# 
#     Lets observe the changes in the coefficients after regularization
# 
#     Q1-Which variables are significant in predicting the price of a house?
#     
#     Q2 -How well those variables describe the price of a house
#     
#     Problem Statement -Part 2
#      
#     Question 1
#     Answer
#     Question 2
#     Answer
#     Question 3
#     Answer
#     Lasso
#     Question 4
#     Answer
# 
# 
# 
# 

# ## 1. Data Overview
# -------------------------------

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


### reading the dataset
HPPrediction_df=pd.read_csv("train.csv")
HPPrediction_df.head()


# In[3]:


## check the dimension
HPPrediction_df.shape


# In[4]:


### check for the column details
HPPrediction_df.info()


# In[5]:


#To check columns present in dataset
print(HPPrediction_df.columns)


# 
# ## 2. Data Cleaning
# ---------
# 
#   - a) Checking and Imputing the null values
#   - b) Outlier Detection and Removal
# 
# 

# In[6]:


### Null Value check
round(HPPrediction_df.isnull().sum()/len(HPPrediction_df.index)*100,2)


# In[7]:


#### null value check
print('Percentage of Missing Values in each column is as follows:')
print(round(HPPrediction_df.isnull().sum()/len(HPPrediction_df.index)*100,2).sort_values(ascending=False)[ round(HPPrediction_df.isnull().sum()/len(HPPrediction_df.index),2) > 0 ] )


# In[8]:


# since 3 columns POOLQC, MiscFeature,Alley have value more than 85% and ID column is not required , so we are dropping the same 
HPPrediction_df.drop(['PoolQC','MiscFeature','Alley','Fence','Id'],axis=1,inplace=True)


# In[9]:


HPPrediction_df.describe()


# In[10]:


print('Percentage of Missing Values in each column is as follows:')
print(round(HPPrediction_df.isnull().sum()/len(HPPrediction_df.index)*100,2).sort_values(ascending=False)[ round(HPPrediction_df.isnull().sum()/len(HPPrediction_df.index),2) > 0 ] )


# In[11]:


##### lets observe the column with highest percentage of missing values
print('The unique values in columns with highest number ')
print('\n')
print('FireplaceQu: ',HPPrediction_df['FireplaceQu'].value_counts())
print('\n')
print('LotFrontage: ',HPPrediction_df['LotFrontage'].value_counts())
print('\n')
print('GarageCond: ',HPPrediction_df['GarageCond'].value_counts())
print('\n')
print('GarageType: ',HPPrediction_df['GarageType'].value_counts())
print('\n')
print('GarageYrBlt: ',HPPrediction_df['GarageYrBlt'].value_counts())
print('\n')
print('GarageFinish: ',HPPrediction_df['GarageFinish'].value_counts())
print('\n')
print('GarageQual: ',HPPrediction_df['GarageQual'].value_counts())
print('\n')
print('BsmtExposure: ',HPPrediction_df['BsmtExposure'].value_counts())
print('\n')
print('BsmtFinType2: ',HPPrediction_df['BsmtFinType2'].value_counts())
print('\n')
print('BsmtFinType1: ',HPPrediction_df['BsmtFinType1'].value_counts())
print('\n')
print('BsmtCond: ',HPPrediction_df['BsmtCond'].value_counts())
print('\n')
print('BsmtQual: ',HPPrediction_df['BsmtQual'].value_counts())
print('\n')
print('MasVnrArea: ',HPPrediction_df['MasVnrArea'].value_counts())
print('\n')
print('MasVnrType: ',HPPrediction_df['MasVnrType'].value_counts())
print('\n')
print('Electrical: ',HPPrediction_df['Electrical'].value_counts())


# ### Missing or null values will be filled with mean, median or mode based on the below condition
# if **numeric** and has skewness then missing value will be filled with **Median** , and if there is **no** **skewness** then it is replaced with **Mean**
# if it is **categorical** then it is replaced with **Mode**
#  
#    #Features	
# 
# GarageYrBlt --- N   skewness is there   so missing value will be filled with Median
# LotFrontag   --N    skewness is there   so missing value will be filled with Median
# 
# ( FireplaceQu -- C, GarageType -- C,GarageCond  --C,GarageQual -- C,GarageFinish -C,BsmtExposure -CC,BsmtFinType2 -C,BsmtFinType1 -C,BsmtConD --   C,BsmtQuaL---  CC,MasVnrArea---   C,MasVnrType---   C,Electrical-- C)
# since the above features are categorical the missing value will be filled with Mode
# 

# In[12]:


for col in ( 'FireplaceQu', 'GarageType','GarageCond','GarageQual','GarageFinish','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType','Electrical'):
    HPPrediction_df[col]=HPPrediction_df[col].fillna(HPPrediction_df[col].mode()[0])


# In[13]:


HPPrediction_df['LotFrontage'].fillna(HPPrediction_df['LotFrontage'].median(),inplace=True)
print(HPPrediction_df.LotFrontage.isnull().sum())
HPPrediction_df['GarageYrBlt'].fillna(HPPrediction_df['GarageYrBlt'].median(),inplace=True)
print(HPPrediction_df.GarageYrBlt.isnull().sum())


# ### **Outlier Detection and Removal**

# In[14]:


def Col_types(dataframe):
    dataframe_numerical=dataframe.select_dtypes(exclude="object")
    dataframe_categorical=dataframe.select_dtypes(include="object")
    
    return dataframe_numerical,dataframe_categorical
continuous_columns,categorical_columns  = Col_types(HPPrediction_df)
print("categorical_columns:=",categorical_columns.columns.tolist(), "\n continuous_columns:=",continuous_columns.columns.tolist())


# In[15]:


def outlier_thresholds(dataframe,col_name, q1=0.05, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    outlier_length=0
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers_values=((dataframe[col_name] < (low_limit)) |(dataframe[col_name] > (up_limit)))
    outlier_length=len(dataframe[outliers_values])
    if dataframe[outliers_values].any(axis=None):
        return True,outlier_length
    else:
        return False,outlier_length
def replace_outlier_threshold(dataframe,col_name,q1,q3):
    low_limit,up_limit=outlier_thresholds(dataframe,col_name,q1,q3)
    dataframe.loc[(dataframe[col_name]<low_limit),col_name]=low_limit
    dataframe.loc[(dataframe[col_name]>up_limit),col_name]=up_limit
     
    


# In[16]:


def detectoutliers():
    for col in continuous_columns:
        print(f"{col} :  {check_outlier(HPPrediction_df, col)}")
detectoutliers()


# #### Note 26 columns have outlier, therefore all the values below or above the outlier will be replaced with min and max value ***
# 

# In[17]:


outlier_cols=['MSSubClass','LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
              'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
              'BsmtFullBath','BsmtHalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
              'Fireplaces','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
              'ScreenPorch','PoolArea','MiscVal','SalePrice']


# In[18]:


for col in outlier_cols:
    replace_outlier_threshold(HPPrediction_df,col,0.05,0.75)
 


# In[19]:


detectoutliers()


# ### Insights on Data Cleaning
#     1. Dropping the below value since it does not add much value to the analysis:
#         3 columns POOLQC, MiscFeature,Alley have value more than 85% and ID column is not required.
#     2. Missing or null values will be filled with mean, median or mode 
#     GarageYrBlt --- N   skewness is there   so missing value will be filled with Median
#     LotFrontag   --N    skewness is there   so missing value will be filled with Median
# 
#      FireplaceQu -- C, GarageType -- C,GarageCond  --C,GarageQual -- C,GarageFinish -C,BsmtExposure -CC,BsmtFinType2 -C,BsmtFinType1 -C,BsmtConD --   C,BsmtQuaL---  CC,MasVnrArea---   C,MasVnrType---   C,Electrical-- C)
#      
#      3.Outlier Detection and Removal
#      
#       26 columns have outlier, therefore all the values below or above the outlier will be replaced with min and max value
#      
#      

# ## 3.Data Visualization

# In[20]:


# sales price correlation matrix
plt.figure(figsize = (16, 10))
n = 25 # number of variables which have the highest correlation with 'Sales price'

corrmat = HPPrediction_df.corr()

cols = corrmat.nlargest(n, 'SalePrice')['SalePrice'].index
sns.heatmap(HPPrediction_df[cols].corr(),annot=True)
plt.show()
# OverallQual,GrLivArea,GarageCars,TotalBsmtSF,GarageArea,1stFlrSF are highly correlated to Saleprice


# ## 4.Data Encoding
# 
#         Dummy variable
# 

# In[21]:


#Categorical columns
HPPrediction_df.select_dtypes(include='object').columns


# In[22]:


# Convert categorical value into Dummy variable
HPPrediction_df=pd.get_dummies(HPPrediction_df,drop_first=True)
HPPrediction_df.head()


# ## 5.Preparing the model

# - 1. Train-Test Split
# - 2. Rescalling the features

# -- Create X and y
# 
# -- Create train and test set(70-30)
# 
# -- Training your model on the training set(learn the coefficient)
# 
# -- Evaluate your model(training set,test set)
# 

# In[23]:


### Create 
y=HPPrediction_df.pop('SalePrice')
y.head()


# In[24]:


y.shape


# In[25]:


X=HPPrediction_df
X.shape


# ##### create train and test split

# In[26]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=50)


# In[27]:


print('X_train shape',X_train.shape)
print('X_test shape',X_test.shape)
print('y_train shape',y_train.shape)
print('y_test shape',y_test.shape)


# ##### Scaling of numeric varaibles 

# In[28]:


X_train.info()


# In[29]:


# columns to be scaled
X_train.select_dtypes(include=['int64','float64']).columns


# In[30]:


num_vars= ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
           'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
           '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
           'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
           'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'MoSold', 'YrSold']


# In[31]:


X_train[num_vars].head()


# ## Rescaling the numeric feature

# In[32]:


# Scaler is applied to all the columns except for the Feature that has 0 or 1(dummy and binary)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[33]:


# Scaler is applied to all the columns except for the Feature that has 0 or 1(dummy and binary)
## fit_transform : will learn xmin and x max as well as it computes(x-xmin)/(xmax-xmin)
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
## transform : will only computes(x-xmin)/(xmax-xmin)
X_test[num_vars] = scaler.transform(X_test[num_vars])
X_train[num_vars].describe()


# ####  all values are between 0 and 1 ,since we applied scaling 

# In[34]:


y_train.head()


# ## 6. Model Building and Evaluation
#          
#          RFE 

# ##### Recursive Feature elimination

# In[35]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[36]:


lm=LinearRegression()


# In[40]:


# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 25)             # running RFE
rfe = rfe.fit(X_train, y_train)


# In[41]:


#Find the top features
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[42]:


col = X_train.columns[rfe.support_]
col
# Extract the top features


# In[43]:


#Check the shape of train and test
X_train1=X_train[col]
X_test1=X_test[col]
print(X_train1.shape)
print(X_test1.shape)
print(y_train.shape)
print(y_test.shape)


# In[44]:


lm1=lm.fit(X_train, y_train)


# In[45]:


# Print the coefficients and intercept
print(lm1.intercept_)
print(lm1.coef_)


# In[46]:


#import libraries for model evalution
from sklearn.metrics import r2_score, mean_squared_error


# In[47]:


#r2score,RSS and RMSE
y_pred_train = rfe.predict(X_train)
y_pred_test = rfe.predict(X_test)

metric = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric.append(mse_test_lr**0.5)


# ## 7.Ridge and Lasso Regression

# In[48]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# ### Ridge Regression

# In[49]:


# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


ridge = Ridge()

# cross validation
folds = 5
ridge_model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
ridge_model_cv.fit(X_train1, y_train) 


# In[50]:


print(ridge_model_cv.best_params_)
print(ridge_model_cv.best_score_)


# #### The optimum value of alpha is 2

# In[57]:


alpha = 2
ridge = Ridge(alpha=alpha)
ridge.fit(X_train1, y_train)
ridge.coef_


# In[58]:


# Lets calculate some metrics such as R2 score, RSS and RMSE
y_pred_train = ridge.predict(X_train1)
y_pred_test = ridge.predict(X_test1)

metric2 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric2.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric2.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric2.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric2.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric2.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric2.append(mse_test_lr**0.5)


# ## Lasso

# In[59]:


lasso = Lasso()

# cross validation
lasso_model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

lasso_model_cv.fit(X_train1, y_train)


# In[60]:


print(lasso_model_cv.best_params_)
print(lasso_model_cv.best_score_)


# ###### The optimum value of alpha is 100

# In[61]:


alpha =100

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train1, y_train) 


# In[62]:


lasso.coef_


# In[63]:


# Lets calculate some metrics such as R2 score, RSS and RMSE
y_pred_train = lasso.predict(X_train1)
y_pred_test = lasso.predict(X_test1)

metric3 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric3.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric3.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric3.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric3.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric3.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric3.append(mse_test_lr**0.5)


# In[64]:


metric2


# In[66]:


# Creating a table which contain all the metrics

lr_table = {'Metric': ['R2 Score (Train)','R2 Score (Test)','RSS (Train)','RSS (Test)',
                       'MSE (Train)','MSE (Test)'], 
        'Linear Regression': metric
        }

lr_metric = pd.DataFrame(lr_table ,columns = ['Metric', 'Linear Regression'] )

rg_metric = pd.Series(metric2, name = 'Ridge Regression')
ls_metric = pd.Series(metric3, name = 'Lasso Regression')

final_metric = pd.concat([lr_metric, rg_metric, ls_metric], axis = 1)

final_metric


# ## 8. Model Evaluation

# The r2_score of lasso is slightly higher than ridge for the test dataset

# In[68]:


ridge_pred = ridge.predict(X_test1)


# In[69]:


# Plotting y_test and y_pred to understand the spread for ridge regression.
fig = plt.figure(dpi=100)
plt.scatter(y_test,ridge_pred)
fig.suptitle('y_test vs ridge_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('ridge_pred', fontsize=16)  
plt.show()


# In[70]:


y_res=y_test-ridge_pred
# Distribution of errors
sns.distplot(y_res,kde=True)
plt.title('Normality of error terms/residuals')
plt.xlabel("Residuals")
plt.show()


# In[71]:


lasso_pred = lasso.predict(X_test1)


# In[72]:


# Plotting y_test and y_pred to understand the spread for lasso regression.
fig = plt.figure(dpi=100)
plt.scatter(y_test,lasso_pred)
fig.suptitle('y_test vs lasso_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('lasso_pred', fontsize=16)  
plt.show()


# In[73]:


y_res=y_test-lasso_pred
# Distribution of errors
sns.distplot(y_res,kde=True)
plt.title('Normality of error terms/residuals')
plt.xlabel("Residuals")
plt.show()


# #### Lets observe the changes in the coefficients after regularization

# In[74]:


betas = pd.DataFrame(index=X_train1.columns)


# In[75]:


betas.rows = X_train1.columns


# In[76]:


betas['Ridge'] = ridge.coef_
betas['Lasso'] = lasso.coef_


# In[77]:


pd.set_option('display.max_rows', None)
betas.head(68)


# The company wants to know:
# 
#     Which variables are significant in predicting the price of a house, and
# 
#     How well those variables describe the price of a house.
# 
# Also, determine the optimal value of lambda for ridge and lasso regression.

# ### Q1-Which variables are significant in predicting the price of a house?

# Answer- The below mentioned variables are significant in predicting the price
# 
# 1.	GrLivArea
# 2.	OverallQual
# 3.	YearBuilt
# 4.	TotalBsmtSF
# 5.	OverallCond
# 6.	LotArea
# 7.	BsmtFinSF1
# 8.	RoofMatl_CompShg
# 9.	SaleType_New
# 10.	Exterior2nd_CmentBd
# 
# 

# ### Q2 -How well those variables describe the price of a house

# Answer-
#                       Ridge Regression                Lasso Regression
# 
# R2 score(Train)--------- 8.79 ----------------------------8.76
# 
# R2 score(Test)-----------8.31-----------------------------8.32

# In[80]:


final_metric


# In[81]:


pd.set_option('display.max_rows', None)
betas.head(68)


# ## Problem Statement -Part 2
# #### Question 1

# ####  What is the optimal value of alpha for ridge and lasso regression? What will be the changes in the model if you choose double the value of alpha for both ridge and lasso? What will be the most important predictor variables after the change is implemented?

# Answer
# 
#     The optimal value of alpha for ridge and lasso regression
# 
#     Ridge Alpha 2
# 
#     lasso Alpha 100
# 
# Ridge Regression

# In[82]:


#Change the alpha value from 2 to 4
alpha = 4
ridge2 = Ridge(alpha=alpha)
ridge2.fit(X_train1, y_train)


# In[87]:


# Lets calculate some metrics such as R2 score, RSS and RMSE
y_pred_train = ridge2.predict(X_train1)
y_pred_test = ridge2.predict(X_test1)

metric2 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric2.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric2.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric2.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric2.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric2.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric2.append(mse_test_lr**0.5)

#Alpha 2
#R2score(train) 8.796016e-01
#R2score(test)  8.311042e-01


# In[93]:


### Lasso Regression
#Change the alpha value from 100 to 200
alpha =200

lasso2 = Lasso(alpha=alpha)
        
lasso2.fit(X_train1, y_train) 


# In[97]:


# Lets calculate some metrics such as R2 score, RSS and RMSE
y_pred_train = lasso2.predict(X_train1)
y_pred_test = lasso2.predict(X_test1)

metric3 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric3.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric3.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric3.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric3.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric3.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric3.append(mse_test_lr**0.5)
#Alpha 100
#R2score(train) 8.759332e-01
#R2score(test)  8.328141e-01


# R2score of training data has decrease and it has increase on testing data for both lasso and Ridge regression

# In[96]:


## important predictor variables
betas = pd.DataFrame(index=X_train1.columns)
betas.rows = X_train1.columns
betas['Ridge2'] = ridge2.coef_
betas['Ridge'] = ridge.coef_
betas['Lasso'] = lasso.coef_
betas['Lasso2'] = lasso2.coef_
pd.set_option('display.max_rows', None)
betas.head(68)


# 
#     LotArea---------------Lot size in square feet
#     OverallQual---------Rates the overall material and finish of the house
#     OverallCond--------Rates the overall condition of the house
#     YearBuilt-------------Original construction date
#     BsmtFinSF1--------Type 1 finished square feet
#     TotalBsmtSF------- Total square feet of basement area
#     GrLivArea-----------Above grade (ground) living area square feet
#     TotRmsAbvGrd----Total rooms above grade (does not include bathrooms)
#     Street_Pave--------Pave road access to property
#     RoofMatl_Metal----Roof material_Metal
# 
# Predictors are same but the coefficent of these predictor has changed

# #### Question 2
# 
# You have determined the optimal value of lambda for ridge and lasso regression during the assignment. Now, which one will you choose to apply and why?
# 
# Answer:
# 
# The r2_score of lasso is slightly higher than lasso for the test dataset so we will choose lasso regression to solve this problem

# #### Question 3
# 
# After building the model, you realised that the five most important predictor variables in the lasso model are not available in the incoming data. You will now have to create another model excluding the five most important predictor variables. Which are the five most important predictor variables now?

# LotArea,OverallQual,YearBuilt,BsmtFinSF1,TotalBsmtSF are the top 5 important predictors.
# GrLivArea,OverallQual,YearBuilt,TotalBsmtSF,OverallCond
# Let's drop these columns

# In[105]:


X_train2 = X_train1.drop(['GrLivArea','OverallQual','YearBuilt','TotalBsmtSF','OverallCond'],axis=1)
X_test2 = X_test1.drop(['GrLivArea','OverallQual','YearBuilt','TotalBsmtSF','OverallCond'],axis=1)


# In[106]:


X_train2.head()


# In[107]:


X_test2.head()


# In[108]:


# alpha 100
alpha =100
lasso21 = Lasso(alpha=alpha)
lasso21.fit(X_train2, y_train) 


# In[110]:


# Lets calculate some metrics such as R2 score, RSS and RMSE
y_pred_train = lasso21.predict(X_train2)
y_pred_test = lasso21.predict(X_test2)

metric3 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric3.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric3.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric3.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric3.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric3.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric3.append(mse_test_lr**0.5) 


# ### R2score of training and testing data has decreased

# In[111]:


#important predictor variables
betas = pd.DataFrame(index=X_train2.columns)
betas.rows = X_train1.columns
betas['Lasso21'] = lasso21.coef_
pd.set_option('display.max_rows', None)
betas.head(68)


# five most important predictor variables
# 
# LotArea
# 
# BsmtFinSF1
# 
# SaleType_New
# 
# RoofMatl_WdShake
# 
# Condition2_PosA
# 
# 
# 

# ### Question 4

# How can you make sure that a model is robust and generalisable? What are the implications of the same for the accuracy of the model and why?

# Answer
# 
# In order to ensure that the test accuracy is equal to the training score, the model needs to be
# generalized. When applied to datasets other than the ones used for training, the model ought to
# yield reliable results. To ensure that the model's predicted accuracy is high, the outliers shouldn't
# be given undue weight. Only those outliers that are pertinent to the dataset should be kept after
# an outliers analysis has been completed to make sure this is not the case. The dataset has to have
# the outliers that don't make sense kept eliminated. A model cannot be relied upon for predictive
# analysis if it lacks robustness.
# 
# 
# The simpler the model, the more resilient and generalizable it will be, even if its accuracy will
# decline. The Bias-Variance trade-off can also be used to understand it. There is more bias but
# less variation and greater generalizability in simpler models. It implies that a reliable and
# generalizable model will function similarly on training and test data, meaning that accuracy will
# not significantly vary between the two sets of data.
# 
# Bias: When a model is unable to adequately learn from the data, it exhibits mistake. When a
# model has a high bias, it cannot extract information from the data. The model does not perform
# well on training or testing data.
# .
# Variance: Variance is error in model, when model tries to over learn from the data. High
# variance means model performs exceptionally well on training data as it has very well trained on
# this of data but performs very poor on testing data as it was unseen data for the model.
# It is important to have balance in Bias and Variance to avoid overfitting and under-fitting of data

# In[ ]:




