# Exercise 1 RFE

# import pandas as pd
# from sklearn.feature_selection import RFE
# from   sklearn.linear_model    import LinearRegression
#
# df = pd.read_csv("/Users/dandidac/Documents/COMP 3948/Data Sets/USA_Housing.csv")
#
# # Seperate the target and independent variable
# x = df.copy()     # Create separate copy to prevent unwanted tampering of data.
# del x['Price']  # Delete target variable.
# del x['Address']
#
# # Target variable
# y = df['Price']
#
# # Create the object of the model
# model = LinearRegression()
#
# # Specify the number of  features to select
# rfe = RFE(model, n_features_to_select=2)
#
# # fit the model
# rfe = rfe.fit(x, y)
# print('\n\nFEATUERS SELECTED\n\n')
# print(rfe.support_)
#
# columns = list(x.keys())
# for i in range(0, len(columns)):
#     if(rfe.support_[i]):
#         print(columns[i])
#
# from sklearn.model_selection import train_test_split
# from sklearn                 import metrics
# import statsmodels.api       as sm
# import numpy                 as np
#
# # Adding an intercept *** This is requried ***. Don't forget this step.
# # The intercept centers the error residuals around zero
# # which helps to avoid over-fitting.
# X = df[['Avg. Area House Age', 'Avg. Area Number of Rooms']]
# X = sm.add_constant(X)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# model = sm.OLS(y_train, X_train).fit()
# predictions = model.predict(X_test) # make the predictions by the model
#
# print(model.summary())
#
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



# Exercise 2 Forward Feature Selection

# import pandas as pd
# from sklearn.feature_selection import RFE, f_regression
# from   sklearn.linear_model    import LinearRegression
#
# df = pd.read_csv("/Users/dandidac/Documents/COMP 3948/Data Sets/USA_Housing.csv")
#
# # Seperate the target and independent variable
# x = df.copy()     # Create separate copy to prevent unwanted tampering of data.
# del x['Price']  # Delete target variable.
# del x['Address']
#
# # Target variable
# y = df['Price']
#
# #  f_regression returns F statistic for each feature.
# ffs = f_regression(x, y)
#
# featuresDf = pd.DataFrame()
# for i in range(0, len(x.columns)):
#     featuresDf = featuresDf._append({"feature":x.columns[i],
#                                     "ffs":ffs[0][i]}, ignore_index=True)
# featuresDf = featuresDf.sort_values(by=['ffs'])
# print(featuresDf)
#
# from sklearn.model_selection import train_test_split
# from sklearn                 import metrics
# import statsmodels.api       as sm
# import numpy                 as np
#
# # Adding an intercept *** This is requried ***. Don't forget this step.
# # The intercept centers the error residuals around zero
# # which helps to avoid over-fitting.
# X = df[['Avg. Area House Age', 'Avg. Area Income']]
# X = sm.add_constant(X)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# model = sm.OLS(y_train, X_train).fit()
# predictions = model.predict(X_test) # make the predictions by the model
#
# print(model.summary())
#
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



# Exercise 3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,\
     precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

PATH = '/Users/dandidac/Documents/COMP 3948/Data Sets/'
FILE = 'framingham_v2.csv'

data = pd.read_csv(PATH + FILE)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.head())
print(data.describe())

def showYPlots(y_train, y_test, title):
    print("\n ***" + title)
    plt.subplots(1,2)

    plt.subplot(1,2 ,1)
    plt.hist(y_train)
    plt.title("Train Y: " + title)

    plt.subplot(1,2, 2)
    plt.hist(y_test)
    plt.title("Test Y: " + title)
    plt.show()

def evaluate_model(X_test, y_test, y_train, model, title):
    showYPlots(y_train, y_test, title)

    preds  = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    print(cm)
    precision = precision_score(y_test, preds, average='binary')
    print("Precision: " + str(precision))

    recall = recall_score(y_test, preds, average='binary')
    print("Recall:    " + str(recall))

    accuracy = accuracy_score(y_test, preds)
    print("Accuracy:    " + str(accuracy))

# Inspect data.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.head())
print(data.describe())

# Impute missing bmi values with average BMI value.
averageBMI = np.mean(data['BMI'])
data['BMI'] = data['BMI'].replace(np.nan, averageBMI)
print(data.describe())

X = data[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
          'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
y = data['TenYearCHD']

# Split the data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Build logistic regressor and evaluate model.
clf = LogisticRegression(solver='newton-cg', max_iter=1000)
clf.fit(X_train, y_train)
evaluate_model(X_test, y_test, y_train, clf, "Before SMOTE")

from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)

clf2 = LogisticRegression(solver='newton-cg', max_iter=1000)
clf2.fit(X_train_SMOTE, y_train_SMOTE)
evaluate_model(X_test, y_test, y_train_SMOTE, clf2, "After SMOTE")

