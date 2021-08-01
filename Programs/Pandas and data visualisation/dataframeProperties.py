import pandas as pd
import numpy as np
import os

#Solution to importing dataset using path error:
# https://stackoverflow.com/questions/37400974/unicode-error-unicodeescape-codec-cant-decode-bytes-in-position-2-3-trunca
#Problem: avoid using normal string as path
os.chdir(r'..\data science\NptelPythonForDataScience\Dataset\data visualisation and Pandas dataset') #change path accordingly
cars_data = pd.read_csv(r'Toyota.csv', index_col=0)

'''
Sometimes we need to work on copy of the original data in order to avoid modification of original data. 
Python provides two type of copying:
	- shallow copy
	- deep copy

ShallowCopy: It is just a new variable, referencing to the original value i.e.,same memory location being accessed 
by two different variables. Any changes made will get reflected to the original value. Two syntax-
	sample = data.copy(deep=False)
	sample= data

DeepCopy: It is completely a new dataframe i.e., new memory location for the new variable. Syntax
	sample = data.copy(deep=True)
'''
cars_data_copy = cars_data.copy(deep=False)
cars_data_dc = cars_data.copy(deep=True)


'''
Attributes of data:
	- dataFrame.index: to get the index(row labels)
	- dataFrame.columns: to get column labels
	- dataFrame.size: to get the total number of elements in the dataFrame
	- dataFrame.shape: to get the dimension
	- dataFrame.memory_usage([index, deep]): the memory usage of each column in bytes
	- dataFrame.ndim: the number of axes/array dimension
'''
cars_index = cars_data.index
print(cars_index)

cars_cols = cars_data.columns
print(cars_cols)

cars_s = cars_data.size
print(cars_s)

cars_sh = cars_data.shape
print(cars_sh)

cars_mem = cars_data.memory_usage()
print(cars_mem)

cars_dim = cars_data.ndim
print(cars_dim)


'''
Indexing:
	- slicing operator '[]'
	- attribute/dot operator '.'

- dataFrame.head([n]): dataFrame.head(6) returns first n rows; by default 5 rows
- dataFrame.tail(n): returns the last n rows

- 'at' and 'iat' to access scalar values
	- at: provides label-based scalar lookups
	dataFrame.at[4, 'FuelType']

	- iat: provides integer-based lookups
	dataFrame.iat[4,5]

- to access a group of rows and columns by label(s), we can use .loc[] as below
	- dataFrame.loc[:, 'FuelType']
	- dataFrame.loc[:, ['FuelType', 'CC']]
'''
print(cars_data.head(4))
print()
print(cars_data.tail(4))
print()
print(cars_data.at[4,'Age'])
print()
print(cars_data.iat[4,2]) #price(0):Age(1):KM(2)
print()
print(cars_data.loc[:,'Age'])
print(cars_data.iloc[:,1:3])

temp = cars_data.loc[:,['Age', 'KM', 'FuelType', 'HP', 'MetColor', 'Automatic', 'CC', 'Doors', 'Weight']]
#let us check if it was a shallow copy or deep copy.
temp.at[0,'Doors'] = temp.at[1,'Doors']
