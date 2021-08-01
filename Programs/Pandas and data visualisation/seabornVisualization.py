
'''
Use of seaborn (which is built upon matplotlib library):
	- Scatter plot
	- Histogram
	- Bar plot
	- Box and whiskers plot
	- Pairwise plot
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(5,4)})
os.chdir(r'..\data science\NptelPythonForDataScience\Dataset\data visualisation and Pandas dataset') #change path accordingly

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=["??", "????"])

#Removing missing values
cars_data.dropna(axis=0, inplace=True)
print(cars_data.info())

'''
Scatter plot:
	- sns.regplot(x,y) #regplot stands for regression plot
	- by default fit_rig = True #so it will plot regression line
	- sns.set(style) #to set background
# xticks and yticks are chosen by default.
# even xlabel and ylabel are taken by default.
    
#other values of style = dark, white, darkgrid, whitegrid
    
the other function to achieve the scatter plot is 
    - sns.lmplot [combines regression plot and faced grid]
    - 'hue' parameter can be used include another variable say z, to show the z types categories with different colors

'''
sns.set(style="darkgrid")
sns.regplot(x=cars_data['Age'], y=cars_data['Price']) #default, fit_reg=True
plt.savefig('snsRegplot.png')
#one of them have to be commented or else both the graphs will get plot on same place

sns.set(style="whitegrid")
sns.regplot(x=cars_data['Age'], y=cars_data['Price'], fit_reg=False, marker="*")

#for regplot it is 'marker'

#Age vs price by FuelType:
# here we will be able to see the distribution of FuelType on Age VS Price
#for lmplot, it is 'markers'

sns.lmplot('Age', 'Price', cars_data, fit_reg=False, markers="*", hue='FuelType', legend=True, palette="Set1")
#color palette is Set1
#plot of x vs y, group by hue
plt.savefig('snslmplot.png')
#legends helps us to identify which color represents which hue category







'''
Histogram using seaborn:
	- histogram in seaborn keeps kernel density estimate (probability density function) by default
	- sns.distplot(x)
'''
sns.distplot(cars_data['Age'])

#without kernel density estimate
sns.distplot(cars_data['Age'], kde=False)

#controlling the number of bins
sns.distplot(cars_data['Age'],kde=False,bins=8)
plt.savefig('snsdistplot.png')

'''
Bar plot:
	- frequency distribution of fuel type of the cars (non-numerical data preferably)
	- sns.countplot(x, data=dataFrame)
'''

sns.set(style="darkgrid")
sns.countplot(x=cars_data['Age'], data=cars_data)
sns.countplot(x=cars_data['FuelType'], data=cars_data)
plt.savefig('snscountplot.png')
#bar plot of one column group by another
#bar plot of x, group by hue.
sns.countplot(x=cars_data['FuelType'], data=cars_data, hue=cars_data['Automatic'])
plt.savefig('snscountplot2.png')
print(np.amax(cars_data['Price']))
print(np.amin(cars_data['Price']))

'''
Box and whiskers plot - [for numerical variable]
	- to visually interpret the five-number summary (mean, median, mode, max., min.)
	- lines: whiskers
	- lower whisker: min. value of price excluding extreme values (1.5 times of q1)
	- upper whisker: max. value of price excluding the extreme values (1.5 times of q3)
	- q1 (lower box line): represents the values that 25% of data belongs to
	- q2 (middle box line): represents the values that 50% of data belongs to [MEDIAN]
	- q3 (upper box line): represents the values that 75% of data belongs to
	- points below and above the two whiskers are called outliers; very few points are there,

sns.boxplot(x)
'''
sns.boxplot(x=cars_data['Price'])

sns.boxplot(y=cars_data['Price'])

'''
Box and whiskers plot - [for numerical data vs categorical data]
'''
sns.boxplot(x=cars_data['FuelType'], y=cars_data['Price'])
plt.savefig('snsboxWhiskers.png')
'''
Box and whiskers plot - [for 3 variables] or Group box and whiskers plot
x vs y group by hue
'''
sns.boxplot(x=cars_data['FuelType'], y=cars_data['Price'], hue=cars_data['Automatic'])
plt.savefig('snscountplot3vars.png')
#alternative syntax
sns.boxplot(x="FuelType", y="Price", hue="Automatic", data=cars_data)

'''
Sometimes we need to plot multiple type of graphs in a single graph
That is when we need to split the plotting window
'''
#plotting together boxplot and histogram
f, (ax_box, ax_hist) = plt.subplots(2, gridspec_kw={"height_ratios":(.25, .75)})
sns.boxplot(x="Price", data=cars_data, ax=ax_box)
sns.distplot(cars_data["Price"], kde=False, ax=ax_hist, bins=8)

#plotting together barplot and boxplot
f, (ax_box, ax_bar) = plt.subplots(2, gridspec_kw={"height_ratios":(.50, .50)})
#f - represents figure
sns.countplot(x=cars_data['FuelType'], ax=ax_bar)
sns.boxplot(x=cars_data['FuelType'],y=cars_data['Price'], ax=ax_box)


'''
Pairwise plots:
	- It is used to plot pairwise relationships in a dataset
	- For all the variables, it creates scatterplots for joint relationships and histograms for univariate distributions
	- sns.pairplot()
	  plt.show()
      
    joint_relationships: when x and y are different variables (will be represented by scatter plots)
    univariate: when both x and y are same variable (will be represented by histogram); the diagonal boxes.
    Only numerical data are considered.
'''
sns.pairplot(cars_data, kind="scatter", hue="FuelType")
#the plot is going to be grouped by hue (different colors for differnt categories of Fueltype)
plt.show()
plt.savefig('snspairwise.png')

#another 
sns.pairplot(cars_data, kind="scatter", hue="Automatic")
plt.show()