# Import the relevant libraries
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


# Load the data
df = pd.read_csv('./dataset/Absenteeism_data.csv')

# Preprocessing data
#Dropping duplicates and irrelevant columns
df = df.drop_duplicates() # Drop duplicated rows
df.reset_index(drop=True, inplace=True) # Reset the dataframe index
df.drop(['ID'], axis=1, inplace=True) # Drop the 'ID' column

# Working on the 'Reasons for Absence' column and obtaining dummies
reasons_column = pd.get_dummies(df['Reason for Absence'], drop_first=True)

# Grouping the reasons
reason_type_1 = reasons_column.loc[:, '1':'14'].max(axis=1) # grouping the first 14 columns and getting the maximum
reason_type_2 = reasons_column.loc[:, '15':'17'].max(axis=1) # grouping the 15th-17th columns and getting the maximum
reason_type_3 = reasons_column.loc[:, '18':'21'].max(axis=1) # grouping the 18th-21st columns and getting the maximum
reason_type_4 = reasons_column.loc[:, '22':].max(axis=1) # grouping the 22nd-28th columns and getting the maximum

# Concatenating the newly created reason columns and dropping the old reason column
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
df = df.drop(['Reason for Absence'], axis=1)

# Creating a varible with column names and renaming the reason columns
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason 1', 'Reason 2', 'Reason 3', 'Reason 4']

df.columns = column_names # Assign column names for the reasons

# Reordering the column names
column_names_reordered_1 = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']

df = df[column_names_reordered_1] # DataFrame of reordered column names

# Converting the 'Date' column to timestamp and separating it by months and and days
df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')

list_months = [date.month for date in df['Date']] # Getting the month values from the 'Date' column
df['Month Value'] = list_months # Create new month column

# Creating a function to output the days of the week from the 'Date' column
def date_to_weekday(day_value):
    return day_value.weekday()

df['Day of the week'] = df['Date'].apply(date_to_weekday) # Creating a new column in the dataframe for the weekdays

df.drop(['Date'], axis=1, inplace=True) # Drop the 'Date' column

# Reordering the column names
column_names_reordered_2 = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Month Value','Day of the week','Transportation Expense', 
        'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education','Children', 'Pets', 'Absenteeism Time in Hours']

df = df[column_names_reordered_2]

# Working on the education column
# Mapping the values as '1' signifies high school while the other 3 signifies education level above high school 
df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})



# Converting the target to output two values where any value greater than the median is excessively late
target = np.where(df['Absenteeism Time in Hours']>df['Absenteeism Time in Hours'].median(), 1, 0)
df['Excessive Absenteeism'] = target # assigning the new target output in the new target column
df = df.drop(['Absenteeism Time in Hours', 'Daily Work Load Average', 
                                           'Distance to Work', 'Day of the week'], axis=1) # Dropping the first target column 

# Select  input features
unscaled_inputs = df.iloc[:, : -1]


# Define the CustomScaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std) # Assigning the scaler object to the instance variable self.scaler
        self.columns = columns # Assigning the list of column names to the instance variable self.columns
        self.mean_ = None # Initialize mean attribute to None for later calculation
        self.var_ = None # Initialize var attribute to None for later calculation

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y) # Fit the %%writefile square_module.pyscaler object to the assigned list of columns among the X columns
        self.mean_ = np.mean(X[self.columns], axis=0) # Compute the mean of the fitted columns column_wise and assign it to the instance variable self.mean
        self.var_ = np.var(X[self.columns], axis=0) # Compute the variance of the fitted columns column_wise and assign it to the instance variable self.var
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns # Specify the initial column order
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns, index=X.index) # Transform the selected columns of input data X in a DataFrame X_scaled
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)] # Select columns from input data X that are not in self.columns in a DataFrame X_not_scaled
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order] # Return the concatenation of the two DataFrame in a specified column order

# Scaling the data
columns_to_scale = ['Transportation Expense', 'Age', 'Body Mass Index','Children', 'Pets', 'Month Value']
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, target, stratify=target, train_size=0.8, random_state=42)

# Train the model
reg = LogisticRegression()
reg.fit(x_train, y_train)



# Predictions and probabilities on the test set
y_pred = reg.predict(x_test)  # Get the predicted classes
y_prob = reg.predict_proba(x_test)[:, 1]  # Get probabilities for the positive class (class 1)

# Calculate precision, recall, and f1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate the AUC (Area Under the Curve)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Checking the accuracies and scores
print(f"Training Score: {reg.score(x_train, y_train):.4f}")
print(f"Test Score: {reg.score(x_test, y_test):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {roc_auc:.4f}")

# Define the file path and save the model
model_file_path = './models/model.pkl'

joblib.dump(reg, model_file_path)