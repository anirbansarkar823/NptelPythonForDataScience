import pandas as pd
import numpy as np
import os

os.chdir(r'..\data science\NptelPythonForDataScience\Dataset\data visualisation and Pandas dataset') #change path accordingly
cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"])

'''
dataFrames stores data in two format:
	- numeric (integer-10:int64, float-10.53:float64)
	- character type (string - 'abc')
    
- base python: int, float
- pandas: int64, float64 ['nan' in numeric makes col dtype to float64]

Character type:
	- category: string variable consisting of few different values. 
	- object: column gets assigned as object type when it holds mixed types. 
    'nan' will definitely lead column to have object character type

'''

'''
dataFrame.dtypes - returns a series with the data type of each column
'''
col_dtype = cars_data.dtypes
print(col_dtype)
car_short = cars_data.loc[:6,:]
col_d_short = car_short.dtypes


'''
Count of unique data types:
	- dataFrame.get_dtype_counts() will return the unique data type count in dataframe; How many columns belongs to 
    which dtype?
'''
# d_count = cars_data.get_dtype_counts() # gives error as it is deprecated since version 0.25.0
d_count = cars_data.dtypes.value_counts()


'''
Selecting data based on data types:
	- pandas.DataFrame.select_dtypes(include=None, exclude=None): selects a subset of columns from dataframe based on column dtype
'''
cars_exclude = cars_data.select_dtypes(exclude=['float64', 'int64'])

'''
Concise summary of dataframe:
	- dataFrame.info(): 
'''
print(cars_data.info())
d_count = cars_data.info()
'''
Unique elements of columns:
	np.unique(array): it cannot be used against the whole dataframe, each column must be passed as array.
'''
ar_HP = cars_data['HP']
#print("the array value is \n", ar_HP)
print(np.unique(cars_data['HP']))
print(np.unique(cars_data['Automatic']))#
print(np.unique(cars_data['Doors']))