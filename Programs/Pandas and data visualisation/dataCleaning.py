import numpy as np
import pandas as pd
import os

os.chdir(r'..\data science\NptelPythonForDataScience\Dataset\data visualisation and Pandas dataset') #change path accordingly
cars_data = pd.read_csv('Toyota.csv',index_col=0)
cars_data_cln = pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"])

print(cars_data.info())
print(cars_data_cln.info())

'''
Converting variable's data types:
	- dataFrame.astype(dtype): used to explicitly convert data types from one to another
'''
#say we want to convert the datatype of column of MetColor from float64 to object
cars_data_cln['MetColor'] = cars_data_cln['MetColor'].astype(object)
cars_data_cln['Automatic'] = cars_data_cln['Automatic'].astype(object)
print(cars_data_cln.info())

'''
Category vs object data type:
	- nbytes() can be used to get the total bytes consumed by the elements of columns
	ndarray.nbytes
'''
print("size of object: ",cars_data_cln['FuelType'].nbytes)
#cars_data_cln['FuelType'] = cars_data_cln['FuelType'].astype('category')
print("size of category: ",cars_data_cln['FuelType'].astype('category').nbytes)
print(cars_data_cln.info())

'''
Cleaing columns:
	- 'Doors' has data ['2', '3', '4', '5', 'five', 'four', 'three'] as unique values
	- we can use the replace() function to replace the values as
dataFrame.replace([to_replace, value, ...])
'to_replace' will get replaced with 'value'

'''
cars_data_cln['Doors'].replace('three', '3', inplace=True) #'three' gets replaced with '3'
cars_data_cln['Doors'].replace('four', '4', inplace=True) #inplace=Ture means data will get updated in dataframe
cars_data_cln['Doors'].replace('five', '5', inplace=True)
print(np.unique(cars_data_cln['Doors']))

cars_data_cln['Doors'] = cars_data_cln['Doors'].astype('int64')
print(cars_data_cln.info())

'''
To detect missing values:
	- dataFrame.isnull: can be used to find the missing values present in columns
    - dataFrame.notnull: can be used to find the non-missing values present in columns
	- True: refers null is there; False: refers null is not there
	- dataFrame.isnull.sum(): will give missing values present in each column
** after finding the missing values, we need to come up with logic to fill those missing values
'''
print("the null values \n", cars_data_cln.isnull().sum())
print()
print("the not-null values \n", cars_data_cln.notnull().sum())
