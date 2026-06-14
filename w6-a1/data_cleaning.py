################ Data cleaning the Iris dataset #################
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# load iris dataset
iris = datasets.load_iris()
# Since this is a bunch, create a dataframe
iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target

iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
#### ===> TASK 1: here - add two more lines of the code to find the number and mean of missing data

# number of missing values per column
missing_values = iris_df.isnull().sum()
print("Number of missing values per column:")
print(missing_values)

# mean of missing values per column
mean_missing = iris_df.isnull().mean()
print("Mean of missing values per column:")
print(mean_missing)

# Remove any empty lines
cleaned_data = iris_df.dropna(how="all", inplace=True)

# Extract first 5 rows of the first 4 columns
iris_X=iris_df.iloc[:5,[0,1,2,3]]
print(iris_X)

# Print target names and first 10 targets
print(iris.target_names)

# Print first 10 targets
print(iris.target[:10])

# Print all targets
print(iris.target[:])

### TASK2: Here - Write a short readme to explain above code and how we can calculate the corrolation amoung featuers with description
# 1. Load the Iris dataset from sklearn datasets
# 2. Create a DataFrame with named columns
# 3. Find the number and mean of missing data
# 4. Remove any empty lines
# 5. Calculate the correlation among features

# Calculate correlation among features
correlation_matrix = iris_df.iloc[:, :4].corr()
print("Correlation matrix:")
print(correlation_matrix)
