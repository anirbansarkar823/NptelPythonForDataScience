import numpy as np
import pandas as pd
import os

'''
Exploratory data analysis:
	- Frequency tables
	- Two-way tables
	- Two-way table: joint probability
	- Two-way table: marginal probability
	- Two-way table: conditional probability
	- Correlation
'''

os.chdir(r'..\data science\NptelPythonForDataScience\Dataset\data visualisation and Pandas dataset') #change path accordingly
cars_d = pd.read_csv('Toyota.csv',index_col=0,na_values=["??","????"])

cars_d_test = cars_d.copy(deep=True) # default

'''
Frequency tables:
	- mainly used to check the relationship between the variables
	- what is the frequency of each categories that are available under a variable
'''

'''
Frequency tables:
	1. pd.crosstab()
	- to compute a simple cross-tabulation of one, two (or more) factors
    
    index:a series/list of series - values to group by in rows
    columns: values to group by in columns
    dropna: not to include columns where whole entry is NaN: it will drop the rows with nan value
'''
crosstab_fuel = pd.crosstab(index=cars_d_test['FuelType'], columns='count', dropna=True)

print(cars_d_test.info())

'''
Two-way tables using pd.crosstab()
'''
crosstab_fA_2way = pd.crosstab(index=cars_d_test['FuelType'], columns=cars_d_test['Automatic'], dropna=True)
#0 - Manual
#1 - Automatic
# dropna=True --> means considering only those rows where 'FuelType' and 'Automatic' none of them are not "nan"
'''
Another way is to use joint probability instead of frequency for Exploratory analysis
Joint Probability: 
    is the likelihood of two independent events happening at the same time
    
	- pd.crosstab()
	index - groupby rows
	columns - groupby columns
	normalize - Ture; converting table values from number to proportion
	dropna
'''
crosstab_jp = pd.crosstab(index=cars_d_test['FuelType'], columns=cars_d_test['Automatic'],normalize=True, dropna=True)

'''
Marginal probability:
	- is the probability of the occurrence of the single event
	- margins = True: will give the row sum and column sum
'''
crosstab_marginP = pd.crosstab(index=cars_d_test['FuelType'], columns=cars_d_test['Automatic'], normalize=True, margins=True, dropna=True)

'''
conditional probability:
	- probability of event (A), given that another event (B) has already occurred

	- pd.crosstab()
	index = dataFrame['col_name']
	columns = dataFrame['col_name']
	margins = True
	dropna = True
    normalize='index' or 'columns' #given that normalize has already occurred
'''
crosstab_marginCP = pd.crosstab(index=cars_d_test['Automatic'], columns=cars_d_test['FuelType'], normalize='index', margins=True, dropna=True)
#probability of variable in columns, given that variable in index have already occurred

crosstab_marginCPc = pd.crosstab(index=cars_d_test['FuelType'], columns=cars_d_test['Automatic'], normalize='columns', margins=True, dropna=True)
#probability of variable in index, given that variable in columns already occurred.

'''
Correlation[MOST IMPT]:
	- the strength of association between two numerical variables (for categorical variable, we can use chi square)
	- [-1, 1] is the range of correlation
	- 1: positively correlated, i.e., on increase of one var, the other also increases
	- (-1): negatively correlated, i.e., on increase of one var, the other decreases
	- 0: refers it is too random, no correlation observed.
    
    dataFrame.corr(self, method='pearson')
	- computes pairwise correlation of columns excluding missing(nan) values
'''

#let us see the correlation among numerical data
num_data = cars_d_test.select_dtypes(exclude=['object'])
corr_matrix = cars_d_test.corr()
print(corr_matrix)

# plotting correlation matrix
import matplotlib.pyplot as plt
plt.matshow(corr_matrix)
plt.show()

# plotting correlation matrix
import seaborn as sns
f, ax = plt.subplots(figsize=(8, 8))
# sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), annot=True, cmap=sns.diverging_palette(220, 10, as_cmap=True),
#             square=True, ax=ax) # this is with different colors
sns.heatmap(corr_matrix, annot=True, ax=ax) #this will show the values
plt.savefig('snsheatmapCorrMatrix.png')
# mask - If passed, data will not be shown in cells where mask is True. Cells with missing values are automatically masked.
# annot - If True, write the data value in each cell.
# cmap -  for colors, The mapping from data values to color space. If not provided, the default will depend on whether center is set.