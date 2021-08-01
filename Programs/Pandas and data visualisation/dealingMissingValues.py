

'''
NaN - will be treated as null values
Pandas provides isnull() and isna() methods to find missing values. 
They returns a dataFrame of boolean values (True: for null value, False: non-null values).
'''

import pandas as pd
import numpy as np
import os

os.chdir(r'..\data science\NptelPythonForDataScience\Dataset\data visualisation and Pandas dataset') #change path accordingly
cars_data = pd.read_csv('Toyota.csv', na_values=["??", "????"], index_col=0)

cars2 = cars_data.copy(deep=True)
cars3 = cars_data.copy(deep=True)

#null values column wise
car_isnull = cars3.isnull().sum()
#car_isna = cars2.isna() --- doesn't works for me

'''
for handling missing values:
    - if a large number of columns are missing: better to get rid of that particular row
    - if 1 or 2 columns are missing: we should try to fill the missing values by predicting based upon pattern
'''

missing = cars3[cars3.isnull().any(axis=1)]
cars3.isnull().any(axis=1)
#any(axis=1) - refers all those rows where, atleast 1 column is nan

'''
Most of time, missing values are completely random.

Two ways of filling missing values:
	- Fill the missing values by mean/median, in case of numerical variable
	- Fill the missing values with the class which has 'maximum count' (mode), in case of categorical variable
        
    Decision of imputing missing values with mean or median:
	- if some values are very large or very low, it will tweak the value to a high or low. So in this case, 
    we should go for median
	- if values are close enough, we should go for mean
	
	- dataFrame.describe()
    25% - this percent of data is less than the corresponding values
    50% - this percent of data is less than the corresponding values MEDIAN[]
    75% - this percent of data is less than the corresponding values
    
    - we can go for mean: if difference between mean and median is <100 (or near)
    - if difference is very high, better to go with the middle value
    
    - to fill missing value: dataFrame.fillna()
'''

cars_desc = cars3.describe()

#As mean and median values are close enought for 'Age' variable, we can replace missing values with mean
print("Mean of Age=",cars3['Age'].mean())
cars3['Age'].fillna(cars3['Age'].mean(), inplace=True)

#The mean and median values differ by almost 5000+ for 'KM' variable, so we will replace missing values with meadian
print("Median of KM=",cars3['KM'].median())
cars3['KM'].fillna(cars3['KM'].median(), inplace=True)

#similarly for variable 'HP', the mean and median values are close, so we can replace missing values with mean
print("Mean of HP=", cars3['HP'].mean())
cars3['HP'].fillna(cars3['HP'].mean(), inplace=True)

#let us see the current status
print()
print(cars3.info())
print()
print(cars3.isnull().sum())

'''
For categorical variable, we will find frequency of each category of the variable
Replacement will be done by highest frequest value (modal value)
we will use, series.value_counts()
By default, value_counts() will exclude nan, and will sort categories as per frequency
'''
print()
print("category wise frequency of variable FuelType")
print(cars3['FuelType'].value_counts())
print(cars3['FuelType'].mode())
#cars3['FuelType'].fillna(cars3['FuelType'].value_counts()[0], inplace=True)
#It will give value, to find index we need to use index[]

cars3['FuelType'].fillna(cars3['FuelType'].value_counts().index[0], inplace=True)
#print(cars3['FuelType'].mode()[0]) could have been also used

print()
print(cars3.isnull().sum())

# although the datatype of MetColor is float64, but its original interpretation is categorical
# it holds the value 1 or 0; which represents 2 categories.
# thus we will use the concept of categorical variable
print(cars3['MetColor'].mode())
cars3['MetColor'].fillna(cars3['MetColor'].mode()[0],inplace=True)
print()
print(cars3.isnull().sum())

'''
#What if we need to fill the missing values in 50+ columns:
    we need to use lambda functions:
    Pandas.apply allow the users to pass a function and apply it on every single value of the Pandas series.
'''
print(cars2.isnull().sum())
print(cars2['KM'].dtype)
cars2['MetColor']= cars2['MetColor'].astype('object')
print(cars2.info())
cars2 = cars2.apply(lambda x:x.fillna(x.mean()) if x.dtype == 'float64' else x.fillna(x.mode()[0]))
#here x represents columns.

print(cars_data.info())
cars_data = cars_data.apply(lambda x:x.astype('float64') if x.dtype in ['float64', 'int64'] else x.astype('object'))
