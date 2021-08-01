import numpy as np
import pandas as pd
import os

os.chdir(r'..\data science\NptelPythonForDataScience\Dataset\data visualisation and Pandas dataset') #change path accordingly

cars_data = pd.read_csv('Toyota.csv',index_col=0,na_values=["??", "????"])

#inserting a new column
cars_data.insert(10, "Price_Class", " ") # dF.insert(index, col_name, value)
cars_dtype = cars_data.dtypes

# this although updates the values, but will through warning: A value is trying to be set on a copy of a slice from a DataFrame
for i in range(0, len(cars_data['Price']), 1):
    if (cars_data['Price'][i] <= 8450): # this way we can access column values
        cars_data['Price_Class'][i] = "Low"
    elif (cars_data['Price'][i] > 11950):
        cars_data['Price_Class'][i] = "High"
    else:
        cars_data['Price_Class'][i] = "Medium"
        
#Let us see if values have been inserted or not
print()
print()
print()
print(np.unique(cars_data['Price_Class'])) #it is not giving frequency of each unique values

#whenever we have series, we can count the value wise frequency using
#Series.value_counts()
print(cars_data['Price_Class'].value_counts())
print(cars_data.isnull().sum())


## to convert Age(given in months) to Age_Y(age in years)
cars_data.insert(11,"Age_Y",0)
cars_dtype = cars_data.dtypes
print(cars_dtype)

def Age_convertor(age):
    y = age/12
    return y

#for i in range(0,len(cars_data['Age']),1):
#    cars_data['Age_Y'][i] = Age_convertor(cars_data['Age'][i])
#    cars_data['Age_Y'][i] = round(cars_data['Age_Y'][i],1)
    
cars_data['Age_Y'] = round(Age_convertor(cars_data['Age']),1) #no need of loop, all the values of series got divided
