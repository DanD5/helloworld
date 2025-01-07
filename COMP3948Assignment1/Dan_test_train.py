import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
import seaborn as sns
from sklearn import metrics
import pickle

# Load and display dataset
PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
CSV_DATA = "grades_V2.csv"
df = pd.read_csv(PATH + CSV_DATA, skiprows=1, encoding="ISO-8859-1", sep=',',
                 names=("school", "sex", "age", "address", "famsize", "Pstatus",
                        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime",
                        "studytime", "failures", "schoolsup", "famsup", "paid", "activities",
                        "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout",
                        "Dalc", "Walc", "health", "absences", "grade"))

# Display initial data and summary
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("Data head:")
print(df.head())
print("\nData description:")
print(df.describe())





# MODEL 1 BEST MODEL                  No binning, no dummies

# Handling missing data by imputing
df.fillna( { "failures" : df["failures"].mean()}, inplace=True)
df.fillna( { "studytime" : df["studytime"].mean()}, inplace=True)

X = df[["Medu", "failures", "age", "goout"]]
X = sm.add_constant(X)

y = df["grade"]

# Initialize KFold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# Lists to store metrics for each fold
rmse_list = []
r2_list = []
adj_r2_list = []
aic_list = []
bic_list = []

# Perform K-Fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training data
    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    predictions = model.predict(sm.add_constant(X_test))

    # Calculate RMSE
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    rmse_list.append(rmse)

    # R^2 and Adjusted R^2
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    r2_list.append(r2)
    adj_r2_list.append(adj_r2)

    # AIC and BIC
    aic = model.aic
    bic = model.bic
    aic_list.append(aic)
    bic_list.append(bic)

# Calculate average metrics across all folds
avg_rmse = np.mean(rmse_list)
avg_r2 = np.mean(r2_list)
avg_adj_r2 = np.mean(adj_r2_list)
avg_aic = np.mean(aic_list)
avg_bic = np.mean(bic_list)

# Display the results
print("K-Fold Cross-Validation Results:")
print(f"Average RMSE: {avg_rmse}")
print(f"Average R^2: {avg_r2}")
print(f"Average Adjusted R^2: {avg_adj_r2}")
print(f"Average AIC: {avg_aic}")
print(f"Average BIC: {avg_bic}")

# Save the model using pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    f.close()

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build and evaluate model
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()  # Fit the model
predictions = model.predict(sm.add_constant(X_test))      # Make predictions using the model

# Print model summary and RMSE
print("\nModel 1 Summary:")
print(model.summary())
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Predicted vs Actual plot
plt.figure(figsize=(12, 6))

# Predicted vs Actual values
plt.subplot(1, 2, 1)
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Residuals vs Actual values plot
residuals = y_test - predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.title('Residuals vs. Actual')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')

plt.tight_layout()
plt.show()

# Correlation Matrix
# Select the relevant columns
correlation_columns = ["Medu", "failures", "goout", "age", "grade"]
correlation_data = df[correlation_columns]

# Compute correlation matrix
correlation_matrix = correlation_data.corr()

# Display matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Correlation Matrix for Model 1 Variables and Grade')
plt.show()





# Model 2                   Binning and dummies

# # # Handling missing data by imputing
# df.fillna( { "failures" : df["failures"].mean()}, inplace=True)
# df.fillna( { "studytime" : df["studytime"].mean()}, inplace=True)
#
# # Binning
# df['absencesBin'] = pd.cut(df['absences'], bins=[-1, 5, 15, 56], labels=['low', 'moderate', 'high'])
# df['ageBin'] = pd.cut(df['age'], bins=[14, 16, 18, 22], labels=['15-16', '17-18', '19+'])
# df['MeduBin'] = pd.cut(df['Medu'], bins=[-1, 1, 3, 4], labels=['low', 'medium', 'high'])
# df['FeduBin'] = pd.cut(df['Fedu'], bins=[-1, 1, 3, 4], labels=['low', 'medium', 'high'])
# df['traveltimeBin'] = pd.cut(df['traveltime'], bins=[0, 2, 4], labels=['short', 'long'])
# df['healthBin'] = pd.cut(df['health'], bins=[0, 2, 3, 5], labels=['low', 'medium', 'high'])
# print(df.head())
#
# # Turning all categorical data into numerical using dummies
# df = pd.get_dummies(df, columns=["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
#                                  "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
#                                  "internet", "romantic", "absencesBin", "ageBin", "MeduBin", "FeduBin", "traveltimeBin",
#                                  "healthBin"],dtype=int)
# print(df.head())
#
# X = df[["ageBin_15-16", "MeduBin_high"]]
# X = sm.add_constant(X)
#
# y = df["grade"]
# # Initialize KFold cross-validation
# kf = KFold(n_splits=5, shuffle=True)
#
# # Lists to store metrics for each fold
# rmse_list = []
# r2_list = []
# adj_r2_list = []
# aic_list = []
# bic_list = []
#
# # Perform K-Fold cross-validation
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     # Fit the model on the training data
#     model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
#     predictions = model.predict(sm.add_constant(X_test))
#
#     # Calculate RMSE
#     rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
#     rmse_list.append(rmse)
#
#     # R^2 and Adjusted R^2
#     r2 = model.rsquared
#     adj_r2 = model.rsquared_adj
#     r2_list.append(r2)
#     adj_r2_list.append(adj_r2)
#
#     # AIC and BIC
#     aic = model.aic
#     bic = model.bic
#     aic_list.append(aic)
#     bic_list.append(bic)
#
# # Calculate average metrics across all folds
# avg_rmse = np.mean(rmse_list)
# avg_r2 = np.mean(r2_list)
# avg_adj_r2 = np.mean(adj_r2_list)
# avg_aic = np.mean(aic_list)
# avg_bic = np.mean(bic_list)
#
# # Display the results
# print("K-Fold Cross-Validation Results:")
# print(f"Average RMSE: {avg_rmse}")
# print(f"Average R^2: {avg_r2}")
# print(f"Average Adjusted R^2: {avg_adj_r2}")
# print(f"Average AIC: {avg_aic}")
# print(f"Average BIC: {avg_bic}")
#
# # Split the dataset into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # Build and evaluate model
# model = sm.OLS(y_train, sm.add_constant(X_train)).fit()  # Fit the model
# predictions = model.predict(sm.add_constant(X_test))      # Make predictions using the model
#
# # Print model summary and RMSE
# print("\nModel 2 Summary:")
# print(model.summary())
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#
# # Predicted vs Actual plot
# plt.figure(figsize=(12, 6))
#
# # Predicted vs Actual values
# plt.subplot(1, 2, 1)
# plt.scatter(y_test, predictions, alpha=0.7)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
# plt.title('Predicted vs. Actual')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
#
# # Residuals vs Actual values plot
# residuals = y_test - predictions
# plt.subplot(1, 2, 2)
# plt.scatter(y_test, residuals, alpha=0.7)
# plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
# plt.title('Residuals vs. Actual')
# plt.xlabel('Actual Values')
# plt.ylabel('Residuals')
#
# plt.tight_layout()
# plt.show()





# MODEL 3                   Only dummy
# Handling missing data by imputing
# df.fillna( { "failures" : df["failures"].mean()}, inplace=True)
# df.fillna( { "studytime" : df["studytime"].mean()}, inplace=True)
#
# # Turning all categorical data into numerical using dummies
# df = pd.get_dummies(df, columns=["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
#                                  "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
#                                  "internet", "romantic"],dtype=int)
#
# print(df.describe())
#
# X = df[["failures", "famsize_LE3", "activities_yes", "nursery_no"]]
# X = sm.add_constant(X)
#
# y = df["grade"]
#
# # Initialize KFold cross-validation
# kf = KFold(n_splits=5, shuffle=True)
#
# # Lists to store metrics for each fold
# rmse_list = []
# r2_list = []
# adj_r2_list = []
# aic_list = []
# bic_list = []
#
# # Perform K-Fold cross-validation
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     # Fit the model on the training data
#     model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
#     predictions = model.predict(sm.add_constant(X_test))
#
#     # Calculate RMSE
#     rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
#     rmse_list.append(rmse)
#
#     # R^2 and Adjusted R^2
#     r2 = model.rsquared
#     adj_r2 = model.rsquared_adj
#     r2_list.append(r2)
#     adj_r2_list.append(adj_r2)
#
#     # AIC and BIC
#     aic = model.aic
#     bic = model.bic
#     aic_list.append(aic)
#     bic_list.append(bic)
#     print(predictions)
#
# # Calculate average metrics across all folds
# avg_rmse = np.mean(rmse_list)
# avg_r2 = np.mean(r2_list)
# avg_adj_r2 = np.mean(adj_r2_list)
# avg_aic = np.mean(aic_list)
# avg_bic = np.mean(bic_list)
#
# # Display the results
# print("K-Fold Cross-Validation Results:")
# print(f"Average RMSE: {avg_rmse}")
# print(f"Average R^2: {avg_r2}")
# print(f"Average Adjusted R^2: {avg_adj_r2}")
# print(f"Average AIC: {avg_aic}")
# print(f"Average BIC: {avg_bic}")
#
# with open('best_model2.pkl', 'wb') as f:
#     pickle.dump(model, f)
#     f.close()
#
# # Split the dataset into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # Build and evaluate model
# model = sm.OLS(y_train, sm.add_constant(X_train)).fit()  # Fit the model
# predictions = model.predict(sm.add_constant(X_test))      # Make predictions using the model
#
# # Print model summary and RMSE
# print("\nModel 3 Summary:")
# print(model.summary())
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#
# # Predicted vs Actual plot
# plt.figure(figsize=(12, 6))
#
# # Predicted vs Actual values
# plt.subplot(1, 2, 1)
# plt.scatter(y_test, predictions, alpha=0.7)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
# plt.title('Predicted vs. Actual')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
#
# # Residuals vs Actual values plot
# residuals = y_test - predictions
# plt.subplot(1, 2, 2)
# plt.scatter(y_test, residuals, alpha=0.7)
# plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
# plt.title('Residuals vs. Actual')
# plt.xlabel('Actual Values')
# plt.ylabel('Residuals')
#
# plt.tight_layout()
# plt.show()

# Create grade categories
bins = [0, 10, 15, 20]
labels = ['Low', 'Medium', 'High']
df['grade_category'] = pd.cut(df['grade'], bins=bins, labels=labels, include_lowest=True)

# Initialize an empty DataFrame to store the counts
count_table = pd.DataFrame()

# Loop through each column and create a crosstab for each variable against grade_category
for column in df.columns:
    if column != 'grade_category':
        crosstab = pd.crosstab(df['grade_category'], df[column], margins=False)
        crosstab.columns = [f"{column}_{val}" for val in crosstab.columns]
        count_table = pd.concat([count_table, crosstab], axis=1)

# Print the count table
print(count_table)
