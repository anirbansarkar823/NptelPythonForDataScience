Note: This Repo is based upon the Introductory course available on NPTEL: "Python for Data Science"
Here, few codes have been written on Jupyter Notebook, and few on spyder.

1. Regression Case Study:
	- first the data is imported with na_values 
	
	- then for each features/variables remove the outliers by plotting and identifying the significant range of values
	
	- plot each variable with the 'price' the output variable to check whether the variable is significant or insignificant. If the feature has the capability to influence the output/dependent variable
	
	- We will remove the insignificant variables in order to avoid overfitting/underfitting
	
	- dataFrame.corr() method can be used to check the correlation among numerical variables
	
	- For Regression we will use two types of models:
		> Linear Regression
		> Random Forest Model
	
	- the dataset for regression will be chosed based upon below two parameters:
		> First dataset: we will remove all the rows which holds missing (NA values, even if a single column in that row has missing value; the row will be dropped)
		> Second dataset: data obtained by imputing missing values with median (for numerical variables) and mode (in case of categorical variable)
	
	- pandas.get_dummies needs to be used to convert all the categorical variables to numerical as ML algorithms works only on numerical value
	
	- # IMPORT NECESSARY LIBRARIES
		from sklearn.model_selection import train_test_split
		from sklearn.linear_model import LinearRegression
		from sklearn.ensemble import RandomForestRegressor
		from sklearn.metrics import mean_squared_error 
	
	- Sometimes some columns holds large values which results is distorted graphs. So to overcome we choose log(column) values

	- metrics for measuring performance 
		> sqrt(mean_squared_error)
		> R-squared value - how close the data is to the regression line - The value varies for  0  to 1; higher value indicates the model has fit better

	- for RandomForestRegressor we need to set the hyperparameters appropriately to avoid overfitting


2. Classification Case Study:

	- As exploratory data analysis:
		> Need to know and understand data
		> Data pre-processing
		> Finally use cross tables (to view the distribution of data between two variable values) and data visualization to see results

	- Sometimes missing values are not in plain 'nan' form. In such cases we need to use numpy methods like values_counts(), unique() to see what values are there. 
		> here missing values were of the form ' ?'
		> data_2 = pd.read_csv('income(1).csv', na_values=[' ?'])

	- Here the dataset is chosen by dropping rows having 'nan' values

	- The relation among variable was visualized using various plots (boxplot, )
	
	- For performance measure we can use 
		> from sklearn.metrics import accuracy_score, confusion_matrix
		> confusion_matrix(test_y, prediction)
		> accuracy_score(test_y, prediction)
	
	- First we used Logistic Regression without removing any features. Then we used Logistic Regression on dataset after removing few insignificant variables

	- Classification was also performed using K - Nearest Neighbours classifiers
		> KNN_classifier = KNeighboursClassifier(n_neighbors=5)

3. Introduction to pandas, matplotlib and Seaborn:

		- ways to import dataset from csv, text and excel files
		- copying dataset (shallow and deep copying)
		- properties of dataFrame.
		- plotting graphs using matplotlib and Seaborn libraries
		- Handling missing values using numpy, pandas

4. Introduction to Jupyter Notebook basics:
	•	Click on spotlight, type terminal to open a terminal window.
	•	Enter the startup folder by typing cd /some_folder_name.
	•	Type jupyter notebook to launch the Jupyter Notebook App The notebook interface will appear in a new browser window or tab.
	•	#<space> any text will increase the size of text.
	•	** Before writing the text, we need to change the type from ‘code’ to ‘markdown’
	•	##<space> a little reduced size then single #
	•	Cell inactive: when blue
	•	Cell active: when green
	•	To add a cell above current cell: press ‘a’ ( after inactivating cell)
	•	To add a cell below current cell: press ‘b’ (after inactivating)
	•	To  delete a cell: press ‘d’ twice (after inactivating)
	•	Extension – ipynb: ipython notebook.

5. Introduction to numpy:
	- various Numpy operations

