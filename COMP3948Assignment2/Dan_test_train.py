import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, f_regression, SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import seaborn as sns

PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
CSV_DATA = "VehicleInsuranceClaims.csv"
df = pd.read_csv(PATH + CSV_DATA, skiprows=1, nrows=5000, encoding="ISO-8859-1", sep=',',
                 names=("Maker", "Model", "Adv_year", "Adv_month", "Color", "Reg_year", "Bodytype",
                        "Runned_Miles", "Engin_size", "Gearbox", "Fuel_type", "Price", "Seat_num",
                        "Door_num", "issue", "issue_id", "Adv_day", "breakdown_date",
                        "repair_complexity", "repair_cost", "repair_hours", "Claim",
                        "category_anomaly", "repair_date"))

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Clean the 'Engin_size' column by removing 'L'
df['Engin_size'] = df['Engin_size'].str.replace('L', '', regex=False).astype(float)

print(df.describe())
df['breakdown_date'] = (df['breakdown_date'].fillna('0/0/0').astype(str).str.replace('/', '', regex=False).astype(int))
df['repair_date'] = (df['repair_date'].fillna('0/0/0').astype(str).str.replace('/', '', regex=False).astype(int))

# Convert categorical columns to dummy variables
df = pd.get_dummies(df, columns=["Maker", "Model", "Color", "Bodytype",
                                         "Gearbox", "Fuel_type", "issue"], dtype=int)

# Handle missing data
df = df.fillna(df.mean(numeric_only=True))





# Average Accuracy: 0.9006
# Average Precision: 0.9035
# Average Recall: 0.9006
# Average F1 Score: 0.8902

# "category_anomaly", "Color_Gelb", "Repair_Hours_Binned_Extreme", "Seat_num", "repair_hours", "repair_cost", "Repair_Cost_Binned_Very High", "Repair_Cost_Binned_High", "Adv_month_Binned_Q3", "Door_num"

# Model 1

# Binning
# bins_price = [0, 5000, 15000, 30000, 100000, float('inf')]
# labels_price = ['Low', 'Medium', 'High', 'Premium', 'Luxury']
# price_binned = pd.cut(df['Price'], bins=bins_price, labels=labels_price)
#
# bins_miles = [0, 50000, 100000, 200000, 500000, float('inf')]
# labels_miles = ['0-50k', '50k-100k', '100k-200k', '200k-500k', '500k+']
# runned_miles_binned = pd.cut(df['Runned_Miles'], bins=bins_miles, labels=labels_miles)
#
# bins_engine = [0, 1.5, 2.5, 3.5, float('inf')]
# labels_engine = ['Small', 'Medium', 'Large', 'Very Large']
# engine_size_binned = pd.cut(df['Engin_size'], bins=bins_engine, labels=labels_engine)
#
# bins_adv_day = [0, 7, 14, 21, 28]
# labels_adv_day = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
# adv_day_binned = pd.cut(df['Adv_day'], bins=bins_adv_day, labels=labels_adv_day)
#
# bins_adv_month = [0, 3, 6, 9, 12]
# labels_adv_month = ['Q1', 'Q2', 'Q3', 'Q4']
# adv_month_binned = pd.cut(df['Adv_month'], bins=bins_adv_month, labels=labels_adv_month, right=False)
#
# bins_adv_year = [2010, 2015, 2020, 2025]
# labels_adv_year = ['2010-2015', '2016-2020', '2021-2025']
# adv_year_binned = pd.cut(df['Adv_year'], bins=bins_adv_year, labels=labels_adv_year)
#
# bins_repair_cost = [0, 500, 1000, 5000, 10000, float('inf')]
# labels_repair_cost = ['Low', 'Medium', 'High', 'Very High', 'Luxury']
# repair_cost_binned = pd.cut(df['repair_cost'], bins=bins_repair_cost, labels=labels_repair_cost)
#
# bins_repair_hours = [0, 1, 3, 5, 10, float('inf')]
# labels_repair_hours = ['Low', 'Medium', 'High', 'Very High', 'Extreme']
# repair_hours_Bbinned = pd.cut(df['repair_hours'], bins=bins_repair_hours, labels=labels_repair_hours)
#
# new_columns = pd.DataFrame({
#     'Price_Binned': price_binned,
#     'Runned_Miles_Binned': runned_miles_binned,
#     'Engine_size_Binned': engine_size_binned,
#     'Adv_day_Binned': adv_day_binned,
#     'Adv_month_Binned': adv_month_binned,
#     'Adv_year_Binned': adv_year_binned,
#     'Repair_Cost_Binned': repair_cost_binned,
#     'Repair_Hours_Binned': repair_hours_Bbinned,
# })
#
# df = pd.concat([df, new_columns], axis=1)
#
# # One-hot encoding
# df = pd.get_dummies(df, columns=['Price_Binned', 'Runned_Miles_Binned', 'Engine_size_Binned',
#                                                  'Adv_day_Binned', 'Adv_month_Binned', 'Adv_year_Binned',
#                                                  'Repair_Cost_Binned', 'Repair_Hours_Binned'], drop_first=True)
#
# # Feature Selection
# X = df.drop(['Claim', "breakdown_date", "repair_date"], axis=1)
# y = df['Claim']
#
# # Scaling
# mmScaler = MinMaxScaler()
# X_mmScaled = mmScaler.fit_transform(X)
#
# # Feature selection
# # RFE
# model = LogisticRegression(max_iter=50000)
# rfe = RFE(model, n_features_to_select=10)
# rfe = rfe.fit(X_mmScaled, y)
# rfe_selected_features_minmax = X.columns[rfe.support_]
#
# print("\n[Min-Max Scaler - RFE Selected Features]")
# print(", ".join(rfe_selected_features_minmax))
#
# # Compute F-statistics and p-values for MinMaxScaler
# minMax_ffs = f_regression(X_mmScaled, y)
# sorted_indices_minmax = np.argsort(minMax_ffs[0])[::-1]
# top_20_indices_minmax = sorted_indices_minmax[:20]
#
# print("\n[Min-Max Scaler - Top 20 Features by F-Statistic]")
# print("{:<30} {:>10}".format("Feature", "F-Statistic"))
# print("-" * 40)
# for i in top_20_indices_minmax:
#     print(f"{X.columns[i]:<30} {minMax_ffs[0][i]:>10.2f}")
#
# # Feature selection using Chi-Square
# chi_test = SelectKBest(score_func=chi2, k=10)
# chi_test.fit(X_mmScaled, y)
# chi_scores = chi_test.scores_
# sorted_indices_chi = np.argsort(chi_scores)[::-1]
# top_20_indices_chi = sorted_indices_chi[:20]
#
# print("\n[Chi-Square - Top 20 Features by Chi-Square Score]")
# print("{:<30} {:>10}".format("Feature", "Chi-Square Score"))
# print("-" * 40)
# for i in top_20_indices_chi:
#     print(f"{X.columns[i]:<30} {chi_scores[i]:>10.2f}")
#
# X = df[["category_anomaly", "Color_Gelb", "Repair_Hours_Binned_Extreme", "Seat_num", "repair_cost"]]
# y = df["Claim"]
#
# kf = KFold(n_splits=10, shuffle=True, random_state=0)
#
# accuracy_scores = []
# precision_scores = []
# recall_scores = []
# f1_scores = []
#
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     X_train_scaled = mmScaler.fit_transform(X_train)
#     X_test_scaled = mmScaler.transform(X_test)
#
#     logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear', random_state=0)
#     logisticModel.fit(X_train_scaled, y_train)
#
#     y_pred = logisticModel.predict(X_test_scaled)
#
#     accuracy_scores.append(accuracy_score(y_test, y_pred))
#     precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
#     recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
#     f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
#
# mean_accuracy = np.mean(accuracy_scores)
# std_accuracy = np.std(accuracy_scores)
#
# mean_precision = np.mean(precision_scores)
# std_precision = np.std(precision_scores)
#
# mean_recall = np.mean(recall_scores)
# std_recall = np.std(recall_scores)
#
# mean_f1 = np.mean(f1_scores)
# std_f1 = np.std(f1_scores)
#
# print(f"Average Accuracy: {mean_accuracy:.4f}")
# print(f"Accuracy Standard Deviation: {std_accuracy:.4f}")
#
# print(f"Average Precision: {mean_precision:.4f}")
# print(f"Precision Standard Deviation: {std_precision:.4f}")
#
# print(f"Average Recall: {mean_recall:.4f}")
# print(f"Recall Standard Deviation: {std_recall:.4f}")
#
# print(f"Average F1 Score: {mean_f1:.4f}")
# print(f"F1 Score Standard Deviation: {std_f1:.4f}")





# Average Accuracy: 0.9440
# Average Precision: 0.9473
# Average Recall: 0.9440
# Average F1 Score: 0.9404

# "category_anomaly", "Color_Gelb", "repair_cost", "repair_hours", "Seat_num", "Price", "Runned_Miles", "Engin_size", "Door_num", "Model_Focus"

# Model 2 BEST MODEL

# Binning
bins_miles = [0, 50000, 100000, 200000, 500000, float('inf')]
labels_miles = ['0-50k', '50k-100k', '100k-200k', '200k-500k', '500k+']
runned_miles_binned = pd.cut(df['Runned_Miles'], bins=bins_miles, labels=labels_miles)

new_columns = pd.DataFrame({
    'Runned_Miles_Binned': runned_miles_binned,
})

df = pd.concat([df, new_columns], axis=1)

# One-hot encoding
df = pd.get_dummies(df, columns=['Runned_Miles_Binned'], drop_first=True)

# Feature Selection
X = df.drop(['Claim'], axis=1)
y = df['Claim']

# RFE
model = LogisticRegression(max_iter=50000, solver='liblinear')
rfe = RFE(model, n_features_to_select=10)
rfe = rfe.fit(X, y)
rfe_selected_features = X.columns[rfe.support_]

print("\n[RFE Selected Features (No Scaling)]")
print(", ".join(rfe_selected_features))

# F-statistics without scaling
f_stats = f_regression(X, y)
sorted_indices_f = np.argsort(f_stats[0])[::-1]
top_20_indices_f = sorted_indices_f[:20]

print("\n[Top 20 Features by F-Statistic (No Scaling)]")
print("{:<30} {:>10}".format("Feature", "F-Statistic"))
print("-" * 40)
for i in top_20_indices_f:
    print(f"{X.columns[i]:<30} {f_stats[0][i]:>10.2f}")

# Chi-Square without scaling
X_shifted = X - X.min().min()

chi_test = SelectKBest(score_func=chi2, k=10)
chi_test.fit(X_shifted, y)
chi_scores = chi_test.scores_
sorted_indices_chi = np.argsort(chi_scores)[::-1]
top_20_indices_chi = sorted_indices_chi[:20]

print("\n[Chi-Square - Top 20 Features by Chi-Square Score (No Scaling)]")
print("{:<30} {:>10}".format("Feature", "Chi-Square Score"))
print("-" * 40)
for i in top_20_indices_chi:
    print(f"{X.columns[i]:<30} {chi_scores[i]:>10.2f}")

X = df[["category_anomaly", "repair_hours", "Seat_num", "Door_num", "Model_Focus"]]
y = df["Claim"]

kf = KFold(n_splits=10, shuffle=True, random_state=0)

final_model = LogisticRegression(fit_intercept=True, solver='liblinear', random_state=0)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average="weighted"))
    recall_scores.append(recall_score(y_test, y_pred, average="weighted"))
    f1_scores.append(f1_score(y_test, y_pred, average="weighted"))


mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

mean_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)

mean_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"Average Accuracy: {mean_accuracy:.4f}")
print(f"Accuracy Standard Deviation: {std_accuracy:.4f}")

print(f"Average Precision: {mean_precision:.4f}")
print(f"Precision Standard Deviation: {std_precision:.4f}")

print(f"Average Recall: {mean_recall:.4f}")
print(f"Recall Standard Deviation: {std_recall:.4f}")

print(f"Average F1 Score: {mean_f1:.4f}")
print(f"F1 Score Standard Deviation: {std_f1:.4f}")

correlations = X.corrwith(y).sort_values(ascending=False)

# HEATMAP
plt.figure(figsize=(6, 8))
sns.heatmap(correlations.to_frame(), annot=True, cmap="coolwarm", cbar=False, fmt=".2f")
plt.title("Correlation Between Features and Claim")
plt.xlabel("Correlation with Claim")
plt.ylabel("Features")
plt.show()

# with open('logistic_model.pkl', 'wb') as file:
#     pickle.dump(final_model, file)





# Average Accuracy: 0.9372
# Average Precision: 0.9372
# Average Recall: 0.9372
# Average F1 Score: 0.9342

# "category_anomaly", "Color_Gelb", "repair_hours", "repair_cost", "Repair_Hours_Binned_Extreme","Repair_Cost_Binned_High", "Seat_num", "Model_Focus", "Repair_Cost_Binned_Very High", "Repair_Cost_Binned_Luxury"

# Model 3

# Binning
# bins_price = [0, 5000, 15000, 30000, 100000, float('inf')]
# labels_price = ['Low', 'Medium', 'High', 'Premium', 'Luxury']
# price_binned = pd.cut(df['Price'], bins=bins_price, labels=labels_price)
#
# bins_miles = [0, 50000, 100000, 200000, 500000, float('inf')]
# labels_miles = ['0-50k', '50k-100k', '100k-200k', '200k-500k', '500k+']
# runned_miles_binned = pd.cut(df['Runned_Miles'], bins=bins_miles, labels=labels_miles)
#
# bins_repair_cost = [0, 500, 1000, 5000, 10000, float('inf')]
# labels_repair_cost = ['Low', 'Medium', 'High', 'Very High', 'Luxury']
# repair_cost_binned = pd.cut(df['repair_cost'], bins=bins_repair_cost, labels=labels_repair_cost)
#
# bins_repair_hours = [0, 1, 3, 5, 10, float('inf')]
# labels_repair_hours = ['Low', 'Medium', 'High', 'Very High', 'Extreme']
# repair_hours_binned = pd.cut(df['repair_hours'], bins=bins_repair_hours, labels=labels_repair_hours)
#
# new_columns = pd.DataFrame({
#     'Price_Binned': price_binned,
#     'Runned_Miles_Binned': runned_miles_binned,
#     'Repair_Cost_Binned': repair_cost_binned,
#     'Repair_Hours_Binned': repair_hours_binned
# })
#
# df = pd.concat([df, new_columns], axis=1)
#
# # One-hot encoding
# df = pd.get_dummies(df, columns=['Price_Binned', 'Runned_Miles_Binned', 'Repair_Cost_Binned', 'Repair_Hours_Binned'], drop_first=True)
#
# # Feature Selection
# X = df.drop(['Claim', "breakdown_date", "repair_date"], axis=1)
# y = df['Claim']
#
# robustScaler = RobustScaler()
# X_robust = robustScaler.fit_transform(X)
#
# # Feature selection using RFE
# model = LogisticRegression(max_iter=50000)
# rfe = RFE(model, n_features_to_select=10)
# rfe = rfe.fit(X_robust, y)
# rfe_selected_features_standard = X.columns[rfe.support_]
#
# print("\n[Standard Scaler - RFE Selected Features]")
# print(", ".join(rfe_selected_features_standard))
#
# # Compute F-statistics and p-values for StandardScaler
# standard_ffs = f_regression(X_robust, y)
# sorted_indices_standard = np.argsort(standard_ffs[0])[::-1]
# top_20_indices_standard = sorted_indices_standard[:20]
#
# print("\n[Standard Scaler - Top 20 Features by F-Statistic]")
# print("{:<30} {:>10}".format("Feature", "F-Statistic"))
# print("-" * 40)
# for i in top_20_indices_standard:
#     print(f"{X.columns[i]:<30} {standard_ffs[0][i]:>10.2f}")
#
# X_robust_non_negative = X_robust - X_robust.min(axis=0)
#
# # Perform Chi-Squared feature selection
# chi_test = SelectKBest(score_func=chi2, k=10)
# chi_test.fit(X_robust_non_negative, y)
#
# chi_scores = chi_test.scores_
# sorted_indices_chi = np.argsort(chi_scores)[::-1]
# top_20_indices_chi = sorted_indices_chi[:20]
#
# print("\n[Chi-Square - Top 20 Features by Chi-Square Score]")
# print("{:<30} {:>10}".format("Feature", "Chi-Square Score"))
# print("-" * 40)
# for i in top_20_indices_chi:
#     print(f"{X.columns[i]:<30} {chi_scores[i]:>10.2f}")
#
#
# # X = df[["category_anomaly", "Color_Gelb", "repair_hours", "repair_cost", "Repair_Hours_Binned_Extreme",
# #                 "Repair_Cost_Binned_High", "Seat_num", "Model_Focus", "Repair_Cost_Binned_Very High", "Repair_Cost_Binned_Luxury"]]
#
# X = df[["category_anomaly", "repair_hours", "Seat_num"]]
# y = df["Claim"]
#
# kf = KFold(n_splits=10, shuffle=True)
#
# accuracy_scores = []
# precision_scores = []
# recall_scores = []
# f1_scores = []
#
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear', random_state=0)
#     X_train, y_train = SMOTE().fit_resample(X_train, y_train)
#     logisticModel.fit(X_train, y_train)
#
#     y_pred = logisticModel.predict(X_test)
#
#     accuracy_scores.append(accuracy_score(y_test, y_pred))
#     precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
#     recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
#     f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
#
# mean_accuracy = np.mean(accuracy_scores)
# std_accuracy = np.std(accuracy_scores)
#
# mean_precision = np.mean(precision_scores)
# std_precision = np.std(precision_scores)
#
# mean_recall = np.mean(recall_scores)
# std_recall = np.std(recall_scores)
#
# mean_f1 = np.mean(f1_scores)
# std_f1 = np.std(f1_scores)
#
# print(f"Average Accuracy: {mean_accuracy:.4f}")
# print(f"Accuracy Standard Deviation: {std_accuracy:.4f}")
#
# print(f"Average Precision: {mean_precision:.4f}")
# print(f"Precision Standard Deviation: {std_precision:.4f}")
#
# print(f"Average Recall: {mean_recall:.4f}")
# print(f"Recall Standard Deviation: {std_recall:.4f}")
#
# print(f"Average F1 Score: {mean_f1:.4f}")
# print(f"F1 Score Standard Deviation: {std_f1:.4f}")