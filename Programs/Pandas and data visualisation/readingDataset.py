import os       #to change working directory if required to import data
import pandas as pd  #to work with data frames

csv_data = pd.read_csv('Iris_data_sample.csv') 
# the data directory and home directory should be same; else error will be thrown
# Blank cells are read as 'nan'
# to assign the first column as index column

csv_data = pd.read_csv('Iris_data_sample.csv', sep=',', index_col=0)
# Junk values can be converted to missing values(i.e., 'nan') by passing them as a list to the parameter 'na_values'
csv_data = pd.read_csv('Iris_data_sample.csv', index_col=0, na_values=["??", "###"])

                                                                       
#Reading excel files/workbood: from specific sheet.
excel_data = pd.read_excel('Iris_data_sample.xlsx', sheetname='Iris_data')
#replacing the junk values 'nan'
excel_data = pd.read_excel('Iris_data_sample.xlsx', sheetname='Iris_data', na_values=["??", "###"], index_col=0)

                                                                                      
#importing text format
text_data = pd.read_table('Iris_data_sample.txt')     
#Problem: all columns read as single column
#delimiter or sep can be used here : they can be "\t", ",", " "
text_data = pd.read_table('Iris_data_sample.txt', sep=" ") 
#text files can also be imported as csv
text_data_as_CSV = pd.read_csv('Iris_data_sample.txt', sep=" ", index_col=0, na_values=["??", "###"])

