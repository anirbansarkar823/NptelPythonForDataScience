# this is the code for case study on classification

#==============================================================================================
# Predicting whether salary is "less than or equal to 50k" or "greater than 50k"
#===============================================================================================

import pandas as pd #to work with data frames
import numpy as np #to work with numerical data

import seaborn as sns #to visualize data
import matplotlib.pyplot as plt
#to partition the data  
from sklearn.model_selection import train_test_split 

#logistic regression
from sklearn.linear_model import LogisticRegression

#importing performance matrics -- accuracy score and confusion metrix
from sklearn.metrics import accuracy_score, confusion_matrix

import os
os.chdir(r'..\data science\NptelPythonForDataScience\Dataset\classification and regression dataset')

data_income = pd.read_csv('income(1).csv')

#creating a copy of the data
data_2 = data_income.copy() #deep copy

'''
Exploratory data analysis:
	1. Getting to know the data (the type of variables)
	2. Data preprocessing (Missing values)
	3. Cross tables and data visualization
'''

'''
------------------------------------------
Getting to know data
------------------------------------------
'''

#data types of each variables
print(data_2.info()) #object - string values; numeric data being represented by int64

#to check the missing data
data_isNull = data_2.isnull() #True-missing data; False-filled data
data_isNullS = data_2.isnull().sum() #no missing data

#to get a descriptive insight - Numerical data
data_des = data_2.describe()
# For variable 'capitalgain' only 25% of data is having value greater than 0
'''
count - total number of observation under the variable
mean - avg. value of the variable
std - 
min - min value observed for the variable
25% - 25% of data under the current variable is less than the specified value
50% - 50% of data under the current variable is less than the specified value
75% - 75%  of data under the current variable is less than the specified value
max - max value observed for the variable
'''

#to include the description of objects:
data_des_O = data_2.describe(include = ['O'])
'''
count - total number of entries for that particular variable
unique - unique categories in that variable
top - the most frequent category
freq - the frequency of top frequent category
'''

print(data_2['JobType'].value_counts())
print(data_2['occupation'].value_counts())

#we can see '?' as missing values
#to find out its original representation
print(np.unique(data_2['JobType']))
print(np.unique(data_2['occupation']))

#' ?' is the representation
# so we again import data, replacing ' ?' as Na
data_2 = pd.read_csv('income(1).csv', na_values=[' ?'])



'''
----------------------------------------
Data Preprocessing
----------------------------------------
'''
#Now let us check for the missing values
data_2.isnull().sum()


# we will subset the rows by considering axis=1
# i.e., considering atleast one column value is missing out of the two
# as a result if 2 columns holds 'nan' values, it too will get considered
missing = data_2[data_2.isnull().any(axis=1)]
#The any() function returns True if any item in an iterable are true, otherwise it returns False.
print(data_2.isnull().any(axis=1))
'''
Points to note:
1. Whenever job-type is 'Never-Worked', occupation is 'nan' (meaningful)
2. Missing values in Jobtype = 1809
3. Missing values in Occupation = 1816
4. There are 1809 rows where two specific columns i.e. occupation & JobType have missing values
5. (1816 - 1809) = 7 => still have occupation unfilled for these 7 rows. Because, jobtype is Never worked
'''

'''
Two choices for missing data:
    - delete the rows holding missing data
    - replace missing cells with alternative values
Now, if data is not missing at random: We have to model the mechanism that produces missing values as well as the 
relationship. This is bit complex.

Here we will remove missing valued rows.
'''
data_2_miss_rem = data_2.dropna(axis=0) #drop all nan holding rows
data_2_miss_rem.isnull().sum()


#Now let us check the correlation among the numerical variable
correlation = data_2_miss_rem.corr()

#let us check the correlation among the categorical variable.
data_2_miss_rem.columns
data_2_miss_rem.index

# gender proportion table
gender = pd.crosstab(index=data_2_miss_rem['gender'], columns='count',normalize=True)

#gender vs salary
gen_sal_stat = pd.crosstab(index=data_2_miss_rem['gender'], columns=data_2_miss_rem['SalStat'], margins=True,normalize='index')

# to see the status of salary visually
salStat_g = sns.countplot(data_2_miss_rem['SalStat']) #seaborn barplot
salStat_d = pd.crosstab(index=data_2_miss_rem['SalStat'], columns='count', normalize=True)


# to see the distribution of age
sns.distplot(data_2_miss_rem['age'], kde=True, bins=10) #seaborn histogram

plt.hist(data_2_miss_rem['age'],color='green',edgecolor='white',bins=20)
plt.xlabel('Age')
plt.ylabel('frequency')
plt.show()

#to see how salary is getting affected by age
sns.boxplot('SalStat','age',data=data_2_miss_rem)


# let us see the employee distribution over JobType vs SalStat
sns.countplot(y=data_2_miss_rem['JobType'],hue=data_2_miss_rem['SalStat'],data=data_2_miss_rem, palette="tab10")
#seaborn barplot

#crosstab to see salary VS job type: 
salStat_Job = pd.crosstab(index=data_2_miss_rem['JobType'], columns=data_2_miss_rem['SalStat'], margins=True, normalize='index')

salStat_Job_100 = pd.crosstab(index=data_2_miss_rem['JobType'], columns=data_2_miss_rem['SalStat'], margins=True, normalize='index').round(4)*100
                              

#Relation between SalType and EdType
sns.countplot(y=data_2_miss_rem['EdType'], hue=data_2_miss_rem['SalStat'],palette="tab10") #seaborn barplot

salStat_Edu = pd.crosstab(index=data_2_miss_rem['EdType'], columns=data_2_miss_rem['SalStat'], normalize='index').round(4)*100


#Relation between SalType and occupation                          
print(np.unique(data_2_miss_rem['occupation']))
sns.countplot(y='occupation', hue='SalStat', data=data_2_miss_rem, palette="tab10")
#seaborn barplot

salStat_occu = pd.crosstab(index=data_2_miss_rem['occupation'], columns=data_2_miss_rem['SalStat'], margins=True, normalize='index').round(4)*100
                           
# Let's explore capitalgain
print(data_2_miss_rem['capitalgain'].value_counts())
# value_counts() reveals that this variable needs to be represented by histogram
sns.set(style="darkgrid")
sns.distplot(data_2_miss_rem['capitalgain'],kde=True, bins=200) # seaborn histogram


#similar graph goes for capitalloss
sns.distplot(data_2_miss_rem['capitalloss'], kde=True, bins=100, color="darkred")


# hoursperweek V/S SalStat: Using box plot
sns.boxplot('SalStat', 'hoursperweek',data=data_2_miss_rem)


'''
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
Case Study on Classification Part II ( LOGISTIC REGRESSION)
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
'''

#Machine Learning models works on numeric variable
# Thus all the categorical variables needs to converted to numeric variable
print(data_2_miss_rem)
data2 = data_2_miss_rem.copy()
print(data2)
print(np.unique(data2['SalStat']))
#reindexing the salary status  names to  0,1
data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})


#one hot encoding: converting various categorical columns.
# each category will get its own columns: 1 in case  category  is present else 0
new_data = pd.get_dummies(data2, drop_first=True)
#drop_first = True -- will drop the first category of each variable/feature

#Now all the columns needs to be divided into columns
# dependent and independent variables
columns_list = list(new_data.columns)
#print(len(columns_list))
#print(len(set(columns_list)))
features = list(set(columns_list) - set(['SalStat']))
print(features)

#storing the output values in y
y = new_data['SalStat'].values
print(y)

#storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train  and test
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2, random_state=0)
#random_state = 0; this implies exery time we run this line for tesing same set of samples will be choosen
# for test and train sets.
# if random_state is given some  value, then chances are more every time we will get a different value

# let us make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x, train_y)

#the model has various attributes, which can be accessed using dot operator
logistic.coef_
logistic.intercept_
logistic.classes_
logistic.__init__
logistic.decision_function
logistic.__le__


#Prediction  from test data
prediction = logistic.predict(test_x)
print(type(prediction))

#confusion matrix
confusion_matrix1 = confusion_matrix(test_y, prediction)
print(confusion_matrix1)
# principle diagonal element gives the count for correct (True positive, True negative)

#accuracy score
accuracy_score1 = accuracy_score(test_y, prediction)
print(round(accuracy_score1*100,2))


#Number  of misclassified samples based upon our trained model
print("Misclassified samples: ", (test_y != prediction).sum())


#=============================================================================
# LOGISTICREGRESSION - Improving accuracy by removing insignificant values
#=============================================================================

data2 = data_2_miss_rem.copy()
#reindexing the salary status  names to  0,1
data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})

insig_cols = ['gender', 'nativecountry','race']
new_data = data2.drop(insig_cols, axis=1)

new_data2 = pd.get_dummies(new_data, drop_first=True)

#the process of finding features
sign_cols = new_data2.columns
print(sign_cols)
features = list(set(sign_cols)-set(['SalStat']))
print(features)

#Storing values 
X = new_data2[features].values
Y = new_data2['SalStat'].values

#splitting into train test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
print(type(X_train))
print(type(X_test))
print(type(Y_train))
print(type(Y_test))
#building mode
logistic = LogisticRegression()
logistic.fit(X_train, Y_train)

#prediction from  test data
prediction1 = logistic.predict(X_test)
print(type(prediction1))
#calculating accuracy
confusion_matrix2 = confusion_matrix(Y_test, prediction1)
accuracy_score2 = accuracy_score(Y_test, prediction1)
print(confusion_matrix1)
print(round(accuracy_score2*100))

#calculating misclassification
print("misclassification observed: ", (Y_test != prediction1).sum())

#==================================================================================================
# KNN
#==================================================================================================
#importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

#import library for plotting
import matplotlib.pyplot as plt

data3 = data_2_miss_rem.copy()
#reindexing the salary status  names to  0,1
data3['SalStat'] = data3['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})

insig_cols = ['gender', 'nativecountry','race']
new_data2 = data3.drop(insig_cols, axis=1)

new_data3 = pd.get_dummies(new_data2, drop_first=True)

cols = new_data3.columns
features = list(set(cols)-set(['SalStat']))

x = new_data3[features]
y = new_data3['SalStat']

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3, random_state=0)


#building the model
KNN_classifier = KNeighborsClassifier(n_neighbors=5)

#Fitting the values for x and y
KNN_classifier.fit(xtrain,ytrain)

# Predicting the test values
prediction3 = KNN_classifier.predict(xtest)

#performance matrix
confusion_matrix3 = confusion_matrix(ytest,prediction3)
accuracy_score3 = accuracy_score(ytest,prediction3)
print(confusion_matrix3)
print(accuracy_score3)

#miscalculated values
print("Miscalculated values: ", (prediction3 != ytest).sum())


#let us experiment on k value on the classifier
accuracy_sample={}
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain,ytrain)
    pred_i  = knn.predict(xtest)
    accuracy_sample[i]=round(accuracy_score(ytest,pred_i)*100)
print(accuracy_sample)
#max is 84.0