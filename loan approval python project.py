# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Prerna\Downloads\Copy of loan - loan.csv")
df.head()
df.tail()
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()

#handling the null values 
# Step 1: Handle missing values in categorical columns
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']

for col in categorical_cols:
    mode_val = df[col].mode()[0]  # Calculate the mode
    df[col].fillna(mode_val, inplace=True)  # Replace missing values with mode

# Step 2: Handle missing values in numeric columns
numeric_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']

for col in numeric_cols:
    median_val = df[col].median()  # Calculate the median
    df[col].fillna(median_val, inplace=True)  # Replace missing values with median
# Verify if there are any remaining missing values
print(df.isnull().sum())


#applicants income-max 80000,min 15 to 10k
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
df.boxplot(column='ApplicantIncome')

# Set title and axis labels
plt.title('Boxplot of Applicant Income')
plt.xlabel('Applicant Income')
plt.ylabel('Income')
plt.xticks(rotation=45)
# Show the plot
plt.show()



# ApplicantIncome grouped by Education
boxplot = df.boxplot(column='ApplicantIncome', by='Education', figsize=(8, 6))

# Set title
plt.title('Boxplot of Applicant Income by Education')

# Set legend
plt.legend(['Not Graduate', 'Graduate'])

# Set axis labels
plt.xlabel('Education')
plt.ylabel('Applicant Income')




#0% of missing geneder
if 'Gender' in df.columns:
    print("per of missing gender is %2f%%" %(df['Gender'].isnull().sum()/df.shape[0]*100))
else:
    print("Gender column does not exist.")


# median  'LoanAmount' column 
# Check if 'LoanAmount' column exists in the DataFrame
if 'LoanAmount' in df.columns:
    # Define exchange rate (1 USD = 75 INR)
    exchange_rate = 75

    # Calculate the median of the 'LoanAmount' column
    median_loan_amount_usd = df['LoanAmount'].median()
    print("Median Loan Amount (USD):", median_loan_amount_usd)

    # Convert median loan amount from USD to INR
    median_loan_amount_inr = median_loan_amount_usd * exchange_rate
    print("Median Loan Amount (INR):", median_loan_amount_inr)
else:
    print("Column 'LoanAmount' not found in the DataFrame")


#average loan amount
# Check if 'LoanAmount' column exists in the DataFrame
if 'LoanAmount' in df.columns:
    # Define exchange rate (1 USD = 75 INR)
    exchange_rate = 75

    # Calculate the mean of the 'LoanAmount' column
    mean_loan_amount_usd = df['LoanAmount'].mean()
    print("Mean Loan Amount (USD):", mean_loan_amount_usd)

    # Convert mean loan amount from USD to INR
    mean_loan_amount_inr = mean_loan_amount_usd * exchange_rate
    print("Mean Loan Amount (INR):", mean_loan_amount_inr)
else:
    print("Column 'LoanAmount' not found in the DataFrame")

    
    
#no. of male and female for loan approval
# Print the number of people who took a loan by gender
print('Number of people who took a loan by gender:')
print(df['Gender'].value_counts())

# Create the count plot
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=df, palette='Set1')

# Add title and formatting
plt.title('Number of People Who Took a Loan by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')

# Show the plot
plt.show()



# Print the number of people who take a loan grouped by marital status
print("Number of people who take a loan as group by marital status:")
print(df['Married'].value_counts())

# Create the count plot
plt.figure(figsize=(8, 6))
sns.countplot(x='Married', data=df, palette='Set1')

# Add title and axis labels
plt.title('Number of People Who Take a Loan by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')

# Show the plot
plt.show()




# Print the number of people who take a loan grouped by number of dependents
print("Number of people who take a loan as group by marital dependents:")
print(df['Dependents'].value_counts())

# Create the count plot
plt.figure(figsize=(8, 6))
sns.countplot(x='Dependents', data=df, palette='Set1')

# Add title and axis labels
plt.title('Number of People Who Take a Loan by Number of Dependents')
plt.xlabel('Number of Dependents')
plt.ylabel('Count')

# Show the plot
plt.show()





## Define the number of bins and bin width for loan amounts
num_bins = 20
bin_width = (df['LoanAmount'].max() - df['LoanAmount'].min()) / num_bins

# Create bins for loan amounts
bins = [i * bin_width for i in range(num_bins+1)]

# Print the number of people who take a loan grouped by loan amount bins
print("Number of people who take a loan as group by LoanAmount:")
print(pd.cut(df['LoanAmount'], bins).value_counts())

## Create the count plot with loan amount bins
plt.figure(figsize=(12, 6))
sns.countplot(x=pd.cut(df['LoanAmount'], bins), palette='Set1')

# Add title and axis labels
plt.title('Number of People Who Take a Loan by Loan Amount')
plt.xlabel('Loan Amount Bins')
plt.ylabel('Count')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
## Show the plot
plt.show()




# Print the number of people who take a loan grouped by credit history
print("Number of people who take a loan as group by marital Credit_History:")
print(df['Credit_History'].value_counts())

# Create the count plot
plt.figure(figsize=(8, 6))
sns.countplot(x='Credit_History', data=df, palette='Set1')

# Add title and axis labels
plt.title('Number of People Who Take a Loan by Credit History')
plt.xlabel('Credit History')
plt.ylabel('Count')
# Show the plot
plt.show()





# Assuming you have already defined df and performed the logarithmic transformation
plt.figure(figsize=(10, 6))
df['loanAmount_log'].hist(bins=20, color='skyblue', edgecolor='black', alpha=0.7)  # Adjust transparency with alpha

# Add title and axis labels
plt.title('Distribution of Logarithmically Transformed Loan Amount', fontsize=16)  # Increase title font size
plt.xlabel('Logarithmically Transformed Loan Amount', fontsize=14)  # Increase x-axis label font size
plt.ylabel('Frequency', fontsize=14)  # Increase y-axis label font size

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.5)  # Add horizontal gridlines with dashed lines

# Show the plot
plt.show()





# Print the number of people who took a loan grouped by self-employment status
print('Number of people who took loan by self employed:')
print(df['Self_Employed'].value_counts())

# Create the count plot
plt.figure(figsize=(8, 6))
sns.countplot(x='Self_Employed', data=df, palette='Set1')

# Add title and axis labels
plt.title('Number of People Who Took a Loan by Self-Employment Status')
plt.xlabel('Self-Employed')
plt.ylabel('Count')

# Show the plot
plt.show()




'''x = df.iloc[:,np.r_[1:5,9:11,13:15]].values
y = df.iloc[:,12].values


X = df.iloc[:, 1:4].values
X = np.concatenate((X, df.iloc[:, 9:10].values), axis=1)
y = np.concatenate((X, df.iloc[:, 13:14].values), axis=1)
print(X)

from sklearn.model_selection import train_test_split

print(len(X))

# Assuming X and y are your feature matrix and target vector respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print(X_train)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Define your model (example: Linear Regression)
model = LinearRegression()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation, adjust cv as needed

# Print the cross-validation scores
print("Cross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())






# Print DataFrame columns
print("DataFrame columns:", df.columns)

# Attempt to access columns
try:
    X = df.iloc[:, np.r_[1:5, 9:11, 13:15]].values
    Y = df.iloc[:, 12].values

    # Print the arrays (X and Y)
    print("X:", X)
    print("Y:", Y)

except IndexError as e:
    print("IndexError:", e)'''
    
import seaborn as sns
import matplotlib.pyplot as plt

# Feature Distribution
sns.pairplot(df.iloc[:, [1, 2, 3, 9, 13]])  # Adjust columns as needed
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.iloc[:, [1, 2, 3, 9, 13]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Selected Features')
plt.show()






# Create a figure and a grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(18, 15))

# Plot 1: Boxplot of Applicant Income
sns.boxplot(data=df, ax=axs[0, 0])
axs[0, 0].set_title('Boxplot of Applicant Income')

# Plot 2: Boxplot of Applicant Income by Education
sns.boxplot(x='Education', y='ApplicantIncome', data=df, ax=axs[0, 1])
axs[0, 1].set_title('Boxplot of Applicant Income by Education')

# Plot 3: Countplot of Gender
sns.countplot(x='Gender', data=df, ax=axs[0, 2])
axs[0, 2].set_title('Count of Loan Applicants by Gender')

# Plot 4: Countplot of Marital Status
sns.countplot(x='Married', data=df, ax=axs[1, 0])
axs[1, 0].set_title('Count of Loan Applicants by Marital Status')

# Plot 5: Countplot of Dependents
sns.countplot(x='Dependents', data=df, ax=axs[1, 1])
axs[1, 1].set_title('Count of Loan Applicants by Number of Dependents')

# Plot 6: Histogram of Logarithmically Transformed Loan Amount
df['loanAmount_log'] = np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins=20, color='skyblue', edgecolor='black', alpha=0.7, ax=axs[1, 2])
axs[1, 2].set_title('Distribution of Logarithmically Transformed Loan Amount')

# Plot 7: Countplot of Self-Employed
sns.countplot(x='Self_Employed', data=df, ax=axs[2, 0])
axs[2, 0].set_title('Count of Loan Applicants by Self-Employment Status')

# Plot 8: Countplot of Credit History
sns.countplot(x='Credit_History', data=df, ax=axs[2, 1])
axs[2, 1].set_title('Count of Loan Applicants by Credit History')

# Remove empty subplot
fig.delaxes(axs[2, 2])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
    






