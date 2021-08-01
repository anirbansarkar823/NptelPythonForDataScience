'''
visualisation makes interpretation of data easier: then normal numerical values

basic plots:
	- scatter plot
        a set of points scattered over a 2 different variable plotted on a horizontal and vertical axes(2D)
        
	- histogram
        > graphical representation using bars of different heights
        > used to represent frequency distribution of a numerical data
        > groups numbers into ranges, and through height we represent the frequency of each range (or bins)
        
	- bar plot
        > to represent categorical data with rectangular bars.
        > the length of each bar represents the frequency of each category of a variable
    
Some of the graphic libraries are:
	- matplotlib: 2D ploting
	- pandas visualization
	- seaborn
	- ggplot
	- plotly: interactive plots
    
'''

#libraries
import pandas as pd #to work on dataFrames
import numpy as np #to work with numerical data
import matplotlib.pyplot as plt #for visualisation
import os #to change file path

#importing data
os.chdir(r'..\data science\NptelPythonForDataScience\Dataset\data visualisation and Pandas dataset') #change path accordingly
cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values = ["??", "????"])

#dropping rows with missing values
cars_data.dropna(axis=0, inplace=True)
print(cars_data.info())

#plotting in a scatter plot: plotting two numeric variables
#plt.scatter(x_axis, y_axis, c='color')
plt.scatter(cars_data['Price'], cars_data['Age'], c='green')
plt.title("graph showing correlation between Price and Age")
plt.xlabel("Price")
plt.ylabel("Age")
plt.show()

#plotting in a Histogram: single variable frequency distribution
plt.hist(cars_data['Age'])
plt.show()

#to make histogram more meaningful
plt.hist(cars_data['Age'], color='green', edgecolor = 'white', bins=8)
plt.title("histogram of Age variable")
plt.xlabel("frequency")
plt.ylabel("Age")
plt.show()

#barPlot
#plt.bar(index, counts, color=['1', '2', '3'])
#plt.xticks(index, labelName,rotation=90 )#rotation specifies the rotation of labels
MtClr_labels = np.unique(cars_data['FuelType'])
indx = np.arange(len(np.unique(cars_data['FuelType'])))
counts = cars_data['FuelType'].value_counts()
print(counts)

#simple bar plot
plt.bar(MtClr_labels, counts)
plt.xticks(indx, MtClr_labels, rotation=75)
#here 0, 200, 400, 600 are yticks
plt.show()

#proper bar plot
plt.bar(indx, counts, color=['green', 'cyan', 'red'])
plt.xticks(indx, MtClr_labels, rotation=75)
plt.xlabel("FuelType")
plt.ylabel("Frequency")
plt.title("Bar plot")