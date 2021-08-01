# this is the code for case study on regression

#==============================================================================================
# Predicting price of pre-owned cars
#===============================================================================================

import pandas as pd
import numpy as np
import seaborn as sns


#==============================================================================================
# Setting dimensions for plot
#==============================================================================================
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.set(rc={'figure.figsize':(5,4)})
#Reading CSV file
import os
os.chdir(r'..\data science\NptelPythonForDataScience\Dataset\classification and regression dataset') #change path accordingly
cars_data = pd.read_csv(r'cars_sampled.csv')

#deep copy
data = cars_data.copy()     

#examining the structure of data
data.info()

#print(np.unique(data['gearbox'])) --> will give error, as nan values present
#print(np.unique(data['abtest'])) --> no error




data.describe()
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#to convert all the displays in .000 format, but just display; if stored in variable, it would still 
# reflect scientific values
carD = data.describe()

#To display maximum set of columns
pd.set_option('display.max_columns',  500)
data.describe()
#data_des_o = data.describe(['O'])


## Dropping unnecessary columns
d_cols = ['dateCrawled', 'dateCreated', 'postalCode', 'lastSeen', 'name']
# we will  remove name, as brand will suffice our work
data = data.drop(d_cols, axis=1)


## dropping duplicate records
data.drop_duplicates(keep='first',  inplace=True)
#  if duplicate records observed, only first occurrences  will be considered


#================================================================================================
# ENTERING INTO  DATA CLEANING
#================================================================================================

#OUTLIERS  MUST BE REMOVED

data.isnull().sum()

'''
Below columns have null values:
    vehicleType            5188
    gearbox                2824
    model                  2758
    fuelType               4503
    notRepairedDamage      9716
'''

print(np.unique(data['yearOfRegistration']))
print(data['yearOfRegistration'].value_counts())

#there are some rubbish years, which needs cleaning
yearwise_count = data['yearOfRegistration'].value_counts().sort_index()
#here we  are sorting index as years have  been placed as index
sum(data['yearOfRegistration'] > 2020) #to  find how many values  are in future
sum(data['yearOfRegistration'] < 1910) #to find cars which are too old
#setting working range for yearOfRegistration 1910 - 2020
sns.regplot(x='yearOfRegistration', y='price', scatter=True, fit_reg=False, data=data)
# seaborn regplot is scatter plots


# not much inferences could be made from the scatter plots as there are lot of outliers.

#price variable: here also  we have outliers which is causing a lot  of skewness towards one  end
price_count = data['price'].value_counts().sort_index()
# index will be price, and their counts  are values
#skewness price=0 has 1400+ values
#sns.set(rc={'figure.figsize':(10,12)})
sns.distplot(data['price'], kde=True, bins=200, color="k") # histogram
data['price'].describe()
sns.boxplot(y=data['price'])
#we will exclude the outliers
sum(data['price'] >150000) #34
sum(data['price']<100) #1778
#thus setting working range  - 100 to 150000 for price


#variable powerPS working range: 10-500
powerPS_count = data['powerPS'].value_counts().sort_index()
print(data['powerPS'].value_counts())
sns.distplot(data['powerPS'], kde=True,bins = 100) #skewness is there: seaborn histogram
data['powerPS'].describe()
sns.boxplot(y=data['powerPS'])
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=data)#scatter plot shows skewness
sns.lmplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=data)
#by trial-and-error let us  set some values
sum(data['powerPS']>500) #only 115
sum(data['powerPS']<10) # 5603



#CLEANING DATA BY removing values not lying in working range
data = data[
        (data.yearOfRegistration <= 2020)
        & (data.yearOfRegistration >= 1950)
        & (data.price >= 100)
        & (data.price <= 15000)
        & (data.powerPS  >= 10)
        & (data.powerPS <= 500)]
#new syntax to drop records/rows: drop rows where features/variables mentioned do  not holds mentioned ranges
# till now 11030 rows have been dropped

# adding meaningful variable to out dataset: Age, using month and  year of registration
data.columns
np.unique(data['monthOfRegistration']) #represents each month in numeric
data['monthOfRegistration'] /= 12
data['monthOfRegistration'] = round(data['monthOfRegistration'],2)
#  array([ 0.  ,  0.08,  0.17,  0.25,  0.33,  0.42,  0.5 ,  0.58,  0.67, 0.75,  0.83,  0.92,  1.  ])

curYear = 2020
data['ageFromRegistration'] = (curYear-data['yearOfRegistration'])+data['monthOfRegistration']

data['ageFromRegistration'].describe()
# mean and median is not much different: implies not skewed

#now dropping the unnecesary variables
d_vars = ['yearOfRegistration', 'monthOfRegistration']
data = data.drop(d_vars, axis=1)




#===========================================================================
# VISUALISING PARAMETERS -  to select features (after selecting working  range)
#===========================================================================

# ageFromRegistration variable
sns.distplot(data['ageFromRegistration'], kde=True, bins=20) #seaborn histogram
sns.boxplot(y=data['ageFromRegistration'])

#price variable
sns.distplot(data['price'],kde=True, bins=100)
sns.boxplot(y=data['price'])

#powerPS variable
sns.distplot(data['powerPS'], kde=True, bins=100)
sns.boxplot(y=data['powerPS'])



#visualisation between variables
#age VS price
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
sns.regplot(x='ageFromRegistration',y='price', scatter=True, fit_reg=True, data=data, ax=ax)#seaborn scatter plot
ax.set(xlim=(0,80)) #to change axis ranges
#ax.set(ylim=(0,30000))
ax.set_ylim(0,30000)
plt.show()
#observation: with increase in age the price mostly decreases

#powerPS VS price
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=True, data=data)
#observation- with increase in powerPS the price is also increasing


#Variable wise cleaning  - REMOVING INSIGNIFICANT CATEGORIES

#variable seller
data['seller'].value_counts()
pd.crosstab(data['seller'], columns='count',  normalize=False)
pd.crosstab(data['seller'], columns='count', normalize=True)#will give  percentage
sns.countplot(x='seller', data=data) #seaborn barplot
#observation: commercial category occupies  only 1 row, thus redundant
# AS  SELLER IS NOT CATEGORICALLY RICH--> INSIGNIFICANT VARIABLE

#variable offerType
data['offerType'].value_counts()
pd.crosstab(data['offerType'],columns='count', normalize=True)
sns.countplot(x='offerType', data=data)
# ONLY ONE CATEGORY IS THERE--> INSIGNIFICANT VARIABLE


#variable abtest 
data['abtest'].value_counts()
pd.crosstab(data['abtest'], columns='count', normalize=True) #bar plot
sns.countplot(x='abtest', data=data)#seaborn barplot
#EQUALLY DISTRIBUTED
sns.boxplot(x='abtest', y='price',data=data)
# for every price  value there  is  almost 50-50 distribution
# So it does not affect price much==> INSIGNIFICANT.


# SOME SIGNIFICANT VARIABLES

# variable vehicleType
data['vehicleType'].value_counts()
pd.crosstab(data['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType', data=data)#seaborn barplot
sns.boxplot(x=data['vehicleType'],  y=data['price'],data=data)
#vehicleType  is a SIGNIFICANT VARIABLE --> as it has  various  categories and all of them 
#  affects the price in various ways.

data.columns
#variable gearbox
data['gearbox'].value_counts()
pd.crosstab(data['gearbox'], columns='count', normalize=True)*100
sns.countplot(x='gearbox', data=data)
sns.boxplot(x='gearbox', y='price',  data=data)
data['gearbox'].describe(include= ['O'])
#gearbox is a SIGNIFICANT VARIABLE  --> althought it have 2  categories  but  each category is affecting
# out dependent variable price


#variable model
data['model'].value_counts().sort_values() #sort the values
pd.crosstab(data['model'], columns='count',normalize=True)*100
sns.countplot(x='model',data=data)#seaborn's barplot
#sns.distplot(data['model'],kde=False) #histogram will not work as it is  not numerical data
sns.boxplot(x='model',  y='price',data=data)
#we will retain the 'model' variable as it  holds various categories:  golf being the dominant

#variable kilometer
data['kilometer'].value_counts()
pd.crosstab(data['kilometer'], columns='count', normalize=True)*100
sns.countplot(x='kilometer', data=data) #barplot in seaborn
sns.boxplot(x='kilometer',y='price', data=data)
sns.distplot(data['kilometer'], kde=True, bins=13)#histogram in seaborn
sns.regplot(x='kilometer', y='price', scatter=True, fit_reg=True, data=data) #scatter  plot in seaborn
#with just a single  anomaly; boxplot clearly  shows that price is  dependent on kilometer variable
# SIGNIFICANT VARIABLE


#variable  fuelType  
data['fuelType'].value_counts()
pd.crosstab(data['fuelType'], columns='count', normalize=True)*100
#sns.countplot(x='fuelType', y='price', data=data) cannot pass values of  both x and y simultaneously
sns.countplot(x='fuelType', data=data)
sns.boxplot(x='fuelType',y='price', data=data)
#clearly  fuelType affects price as various categories of fuelType gives different prices
# SIGNIFICANT VARIABLE

data.columns
#variable brand
data['brand'].value_counts()
pd.crosstab(data['brand'], columns='count', normalize=True)*100
sns.countplot(y='brand', data=data)
sns.boxplot(x='brand', y='price',data=data)
#boxplot  makes  it extremely clear that price is highly dependent on brand. 
#Brands like porche has higher median value  in relation to price
#SIGNIFICANT VARIABLE


#one more important variable is notRepairedDamage
# yes - car is currently in damaged state and has not been rectified
# no - car was damaged but has  also been rectified
data['notRepairedDamage'].value_counts()
pd.crosstab(data['notRepairedDamage'], columns='count', normalize=True)*100
sns.countplot(x='notRepairedDamage', data=data)
sns.boxplot(x='notRepairedDamage',  y='price', data=data)
#boxplot clearly shows  that, car where damage  has been replaired  (no) is having higher median value  
# with respect to price

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(rc={'figure.figsize':(5,4)})
sns.set(rc={'figure.figsize':(25,10)})



# ====================================================================================================
# REMOVING INSIGNIFICANT VARIABLES
# ====================================================================================================
d_cols= ['seller', 'offerType', 'abtest']#list of dropping variables
data = data.drop(d_cols,  axis=1)
#storing  in new  variable  using deep copy
data_C = data.copy()



# ======================================================================================================
# CORRELATION
#=======================================================================================================
# to see the correlation  between numerical variables  
data_C.corr() #it will directly choose the numerical variable

# the other method is to feed only numeric variables
cars_select = data_C.select_dtypes(exclude=[object])
corr_num = cars_select.corr()
round(corr_num,3)

#price column's correlation with other numeric columns
print(corr_num.loc[:,'price'].abs().sort_values(ascending=False)[1:])
# loc- helps to choose columns by  their  name
# abs() is used for dataframe

#============================================================================
'''
we will use two types of models 
    Linear Regression
    Random Forest model

We will use two sets of data
    data obtained from removing rows containing even  a single  missing value
    data  obtained by imputing the missing values
'''

#==========================================================================
# OMITTING MISSING VALUES
#=========================================================================
cars_omit = data_C.dropna(axis=0) #removing missing valued rows

# MODEL CREATION
# creating dummies  for  categorical variable
cars_omit  = pd.get_dummies(cars_omit, drop_first=True)


# IMPORTING NECESSARY LIBRARIES
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Separating input and output features
x1 = cars_omit.drop(['price'], axis='columns', inplace=False)
y1 = cars_omit['price']


# The logarithm in base e is the natural logarithm: np.log(y1)
# plotting the price with log(price)
prices = pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
prices.hist(bins=10)
# This function calls matplotlib.pyplot.hist(), on each series in the DataFrame, resulting in one histogram per column.

#transforming price to logarithmic value to avoid huge ranges
y1 = np.log(y1)


#train test split
X_train, X_test, Y_train, Y_test = train_test_split(x1, y1, test_size=0.3, random_state=1)
#random_state = intValue; every time we run the same algorithm, the same set of records will go to train and test
print(X_train)
print(cars_omit.shape)

# ==========================================================================
# BASELINE MODEL FOR OMITTED DATA
# ==========================================================================

'''
we are making a base model by using test data mean value
This  is to set a benchmark and to compare with our regression model
'''

# finding the mean for test data values
base_pred = np.mean(Y_test)
print(base_pred)

# Repeating the same value till length of the test data
# np.repeat(x,repeats, axis)
base_pred =  np.repeat(base_pred, len(Y_test))

# finding the RMSE(root mean  square  error)
#  is sqrt(summation((Y_test)^2 - (predicted  Values)^2)/number of terms)

base_root_mean_square_error = np.sqrt(mean_squared_error(Y_test, base_pred))
print(base_root_mean_square_error) #0.986188354869

#Now our objective is to build  models whose RMSE value is should be less than this

#=================================================================================================
# LINEAR REGRESSION WITH OMITTED DATA
#=================================================================================================

# The intercept (often labeled as constant) is the point where the function crosses the y-axis
# a regression without a constant (intercept) means that the regression line goes through the origin 
# will set intercept as true
lgr = LinearRegression(fit_intercept=True)

#model
model_lin1 = lgr.fit(X_train,Y_train)
print(model_lin1)

#  predicting model on test set
cars_predict_linR1 = lgr.predict(X_test)

# computing MSE and RMSE
linR_mse1 = mean_squared_error(Y_test, cars_predict_linR1)
linR_rmse1 = np.sqrt(linR_mse1)
print(linR_rmse1) #0.526617468925



# R-squared value
# R-squared is a statistical measure of how close the data are to the fitted regression line
# The value varies for  0  to 1.
# higher values indicates the model has better  fitted the data, model was able to explain all variability of our
# response (dependent) data around its mean, points are closer to regression line
r2_linR_test1 = model_lin1.score(X_test, Y_test) #test values together
r2_linR_train1 = model_lin1.score(X_train, Y_train) #train values together
print(r2_linR_test1) #0.714851702564
print(r2_linR_train1) # 0.722351030384

# Regression diagnostic - Residual plot analysis
residuals1 = Y_test - cars_predict_linR1
sns.regplot(x=cars_predict_linR1, y=residuals1, scatter=True, fit_reg=True)
sns.regplot(x=cars_predict_linR1, y=Y_test, scatter=True, fit_reg=True)
residuals1.describe()# mean=0.002 which shows the Y_test and predicted values  are very close

lgr2 = LinearRegression(fit_intercept=False)
lgr2.fit(X_train, Y_train)
cars_predict_linR2 = lgr2.predict(X_test)
linR_rmse2 = np.sqrt(mean_squared_error(Y_test, cars_predict_linR2))
print(linR_rmse2)#0.567872603339 more when compared to fit_intercept=True
sns.regplot(x=cars_predict_linR2, y=Y_test, scatter=True, fit_reg=True)


# ==========================================================================================
# RANDOM FOREST WITH OMITTED  DATA
# ===========================================================================================

#random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction.

#Model Parameters
# all these parameters are called hyperparameters
# keeping them to default values would result in overfitting in case of decision trees or RandomForests
# the number of trees is to reduce impurity (Gini is a measure of impurity.) until the nodes contain values of same category (gini=0)
# so the main motive  behind a split is to decrease impurity with every split.
# The other function to evaluate the quality of a split is entropy which is a measure of uncertainty or randomness.
# We do not want a tree with all pure leaf nodes (gini=0). It would be too specific and likely to overfit.
# this can be  achieved by setting various hyperparameters to non-default values like max_depth, min_samples_split.

# https://builtin.com/data-science/random-forest-algorithm
# Random forest adds additional randomness to the model, 
# Instead of searching for the most important feature while splitting a node, it searches for the best feature among a random subset of features.
# in random forest, only a random subset of the features is taken into consideration by the algorithm for splitting a node.
# So among all the  trees  created by choosing random  subsets of  features, the output which got  recommended  the most
# will  be  the final answer.
rf = RandomForestRegressor(n_estimators=100, max_features='auto',\
                           max_depth=100, min_samples_split=10,\
                           min_samples_leaf=4,random_state=1)

# model
model_rf1 = rf.fit(X_train, Y_train)

# Predicting  model on test  set
cars_predictions_rf1 = rf.predict(X_test)

# computing MSE and RMSE
rf_mse1 = mean_squared_error(Y_test, cars_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1) #0.420558528244 [best rmse]
# lower error implies better predictions

# computing R  squared value
r2_rf_test1 = model_rf1.score(X_test, Y_test)
r2_rf_train1 = model_rf1.score(X_train, Y_train)
print(r2_rf_train1, r2_rf_test1) # 0.891471388653 0.818141691038
# higher values implies better model  fit


# =================================================================================
# MODEL BUILDING WITH IMPUTED DATA
# ==================================================================================

# we will not drop the na holding coluns this time,  instead update them  with median(numeric variable) and 
# mode(max. frquency for categorical variables)

# DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwds)
# applies func  along an axis in the dataframe
# axis= 0: apply function to each columns
# axis = 1: apply func  to  each rows
# by-dafault, axis =0
cars_imputed = data.apply(lambda x:x.fillna(x.median()) \
                          if  x.dtype  == 'float' else \
                          x.fillna(x.value_counts().index[0]))


cars_imputed.isnull().sum()
data.isnull().sum()

# Converting categorical variable to dummy variables
cars_imputed = pd.get_dummies(cars_imputed, drop_first=True)


# ============================================================
# MODEL BUILDING WITH IMPUTED DATA
# ============================================================

# seperating input  and output features
x2 = cars_imputed.drop(['price'], axis='columns', inplace=False)
y2 = cars_imputed['price']

# plotting the variable  price
# normalval VS log value
prices = pd.DataFrame({"1. before":y2, "2. After":np.log(y2)})
prices.hist() 

# logarithmic values are giving more bell  shaped graph
# thus transforming y2 to logarithmic  form
y2 = np.log(y2)
x_train, x_test, y_train, y_test = train_test_split(x2,y2,test_size=0.3,random_state=1)


## BASELINE MODEL FOR  IMPUTED DATA
'''
The base model is being built using the test data mean value
This is to set a benchmark and to compare with out regression model later
'''

base_pred = np.mean(y_test)
base_pred = np.repeat(base_pred,  len(y_test)) # to repeat same  value and makeit  of same size as y_test


# finding the RMSE
base_root_mean_square_error_imputed = np.sqrt(mean_squared_error(y_test, base_pred))
print(base_root_mean_square_error_imputed) #1.03840779263


## LINEAR REGRESSION WITH IMPUTED DATA

#setting intercept as true
lgr2 = LinearRegression(fit_intercept=True)

# model
model_lin2 = lgr2.fit(x_train,y_train)

#  predicting model on test set
cars_predictions_lin2 = lgr2.predict(x_test)

# computing MSE and RMSE
lin_mse2 = mean_squared_error(y_test, cars_predictions_lin2)
lin_rmse2 = np.sqrt(lin_mse2)
print(lin_rmse2) #0.654839150473 --> less error is good

# R squared  value
# how close the values  are to regression line
# higher values=> better fit
r2_lin_test2 = model_lin2.score(x_test, y_test)
r2_lin_train2 = model_lin2.score(x_train, y_train)
print(r2_lin_train2, r2_lin_test2)
# 0.609062266584 0.602320324587 --> greater is good

# ==================================================================
## RANDOM FOREST WITH IMPUTED DATA
# ==================================================================

# model parameters
# n_estimators - number of trees in  the forest
rf2 = RandomForestRegressor(n_estimators = 100, max_features='auto',
                            max_depth=100, min_samples_split=10,
                            min_samples_leaf=4, random_state=1)

# model
model_rf2 = rf2.fit(x_train, y_train)

# Predicting model on test set
cars_predictions_rf2 = rf2.predict(x_test)

# computing MSE and RMSE
rf_mse2 = mean_squared_error(y_test, cars_predictions_rf2)
rf_rmse2 = np.sqrt(rf_mse2)
print(rf_rmse2) # less is better - 0.576893478096

# R squared value
r2_rf_test2 = model_rf2.score(x_test, y_test)
r2_rf_train2 = model_rf2.score(x_train, y_train)
print(r2_rf_test2, r2_rf_train2) # the more, the better - 0.691357746521 0.808288948336






 





