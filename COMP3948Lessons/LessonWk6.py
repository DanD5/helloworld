#Example 1 / 3 MinMaxScaler

# import pandas  as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
#
# PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "computerPurchase.csv"
# df = pd.read_csv(PATH + CSV_DATA)
#
# # Separate into x and y values.
# X = df[["Age", "EstimatedSalary"]]
# y = df['Purchased']
#
# ## SECTION A ########################################
# from sklearn.preprocessing import MinMaxScaler
# X_train, X_test, y_train, y_test = train_test_split(
#                             X, y, test_size=0.25)
#
# sc_x            = MinMaxScaler()
# X_train_scaled  = sc_x.fit_transform(X_train) # Fit and transform X.
# X_test_scaled   = sc_x.transform(X_test)      # Transform X.
# ## SECTION A ########################################
#
# ## SECTION B ########################################
# # Perform logistic regression.
# logisticModel = LogisticRegression(fit_intercept=True,
#                                    solver='liblinear')
# # Fit the model.
# logisticModel.fit(X_train_scaled, y_train)
# y_pred        = logisticModel.predict(X_test_scaled)
# ## SECTION B ########################################
#
# # Show confusion matrix and accuracy scores.
# cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
#
# print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix")
# print(cm)



# Example 4 StandardScaler

# import pandas  as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
#
# PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "computerPurchase.csv"
# df = pd.read_csv(PATH + CSV_DATA)
#
# # Separate into x and y values.
# X = df[["Age", "EstimatedSalary"]]
# y = df['Purchased']
#
# ## SECTION A ########################################
# from sklearn.preprocessing import StandardScaler
# X_train, X_test, y_train, y_test = train_test_split(
#                             X, y, test_size=0.25)
#
# sc_x            = StandardScaler()
# X_train_scaled  = sc_x.fit_transform(X_train) # Fit and transform X.
# X_test_scaled   = sc_x.transform(X_test)      # Transform X.
# ## SECTION A ########################################
#
# ## SECTION B ########################################
# # Perform logistic regression.
# logisticModel = LogisticRegression(fit_intercept=True,
#                                    solver='liblinear')
# # Fit the model.
# logisticModel.fit(X_train_scaled, y_train)
# y_pred        = logisticModel.predict(X_test_scaled)
# ## SECTION B ########################################
#
# # Show confusion matrix and accuracy scores.
# cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
#
# print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix")
# print(cm)





# import pandas  as pd
# PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "computerPurchase.csv"
# df       = pd.read_csv(PATH + CSV_DATA)
#
# # Separate into x and y values.
# X = df[["Age", "EstimatedSalary"]]
# y = df['Purchased']
#
# # Split data.
# from sklearn.preprocessing import StandardScaler
# import numpy as np
#
# def showAutomatedScalerResults(X):
#     sc_x         = StandardScaler()
#     X_Scale      = sc_x.fit_transform(X)
#     salary       = X.iloc[0][1]
#     scaledSalary = X_Scale[0][1]  # Get first scaled salary.
#     print("The first unscaled salary in the list is: " + str(salary))
#     print("$19,000 scaled using StandardScaler() is: " + str(scaledSalary))
#
# def getSD_with_zeroDegreesFreedom(X):
#     mean = X['EstimatedSalary'].mean()
#
#     # StandardScaler calculates the standard deviation with zero degrees of freedom.
#     s1 = df['EstimatedSalary'].std(ddof=0)
#     print("sd with 0 degrees of freedom automated: " + str(s1))
#
#     # This is the same calculation manually. (**2 squares the result)
#     s2 = np.sqrt(np.sum(((X['EstimatedSalary'] - mean) ** 2)) / (len(X)))
#     print("sd with 0 degrees of freedom manually:  " + str(s2))
#
#     return s1
#
# print("*** Showing automated results: ")
# showAutomatedScalerResults(X)
#
# print("\n*** Showing manually calculated results: ")
# sd     = getSD_with_zeroDegreesFreedom(X)
# mean   = df['EstimatedSalary'].mean()
# scaled = (19000 - mean) / sd
#
# print("$19,000 scaled manually is: " + str(scaled))



# Example 6 RobustScaler

# import pandas  as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
#
# PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "computerPurchase.csv"
# df = pd.read_csv(PATH + CSV_DATA)
#
# # Separate into x and y values.
# X = df[["Age", "EstimatedSalary"]]
# y = df['Purchased']
#
# ## SECTION A ########################################
# from sklearn.preprocessing import RobustScaler
# sc_x    = RobustScaler()
# X_Scale = sc_x.fit_transform(X)
#
# # Split data.
# X_train, X_test, y_train, y_test = train_test_split(
#     X_Scale, y, test_size=0.25, random_state=0)
#
#
# X_train_scaled  = sc_x.fit_transform(X_train) # Fit and transform X.
# X_test_scaled   = sc_x.transform(X_test)      # Transform X.
# ## SECTION A ########################################
#
# ## SECTION B ########################################
# # Perform logistic regression.
# logisticModel = LogisticRegression(fit_intercept=True,
#                                    solver='liblinear')
# # Fit the model.
# logisticModel.fit(X_train_scaled, y_train)
# y_pred        = logisticModel.predict(X_test_scaled)
# ## SECTION B ########################################
#
# # Show confusion matrix and accuracy scores.
# cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
#
# print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix")
# print(cm)



# Example 7 cross fold validation

# # scikit-learn k-fold cross-validation
# from numpy import array
# from sklearn.model_selection import KFold
#
# # data sample
# data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
#
# # splits data into 4 randomized folds
# kfold = KFold(n_splits=4, shuffle=True)
#
# # enumerate splits
# for train, test in kfold.split(data):
#     print('train: %s, test: %s' % (data[train], data[test]))



# Example 8 cross fold validation for logistic regression

# import pandas as pd
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# import numpy as np
# from sklearn.metrics import classification_report, roc_auc_score, precision_score
#
# # Load data
# PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "computerPurchase.csv"
# df = pd.read_csv(PATH + CSV_DATA, sep=',')
#
# # Prepare K-Fold cross-validation with three folds
# kfold = KFold(n_splits=3, shuffle=True, random_state=1)
# accuracyList = []
# precisionList = []
# recallList = []
# foldCount = 1
#
#
# def getTestAndTrainData(trainIndexes, testIndexes, df):
#     dfTrain = df.iloc[trainIndexes, :]  # Gets all rows with train indexes.
#     dfTest = df.iloc[testIndexes, :]  # Corrected to use test indexes.
#     X_train = dfTrain[['EstimatedSalary', 'Age']]
#     X_test = dfTest[['EstimatedSalary', 'Age']]
#     y_train = dfTrain[['Purchased']]
#     y_test = dfTest[['Purchased']]
#     return X_train, X_test, y_train, y_test
#
#
# # Choose scaler type here
# scaler_type = 'Robust'
#
# for trainIdx, testIdx in kfold.split(df):
#     X_train, X_test, y_train, y_test = getTestAndTrainData(trainIdx, testIdx, df)
#
#     # Scaling
#     if scaler_type == 'MinMax':
#         scaler = MinMaxScaler()
#     elif scaler_type == 'Standard':
#         scaler = StandardScaler()
#     elif scaler_type == 'Robust':
#         scaler = RobustScaler()
#     else:
#         raise ValueError("Invalid scaler type")
#
#     X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform.
#     X_test_scaled = scaler.transform(X_test)  # Transform only.
#
#     # Perform logistic regression
#     logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear')
#     logisticModel.fit(X_train_scaled, y_train.values.ravel())  # Flatten y_train.
#     y_pred = logisticModel.predict(X_test_scaled)
#     y_prob = logisticModel.predict_proba(X_test_scaled)
#
#     # Show accuracy scores and other metrics
#     accuracy = metrics.accuracy_score(y_test, y_pred)
#     accuracyList.append(accuracy)
#
#     # Precision and Recall
#     precision = precision_score(y_test, y_pred)
#     precisionList.append(precision)
#     recall = metrics.recall_score(y_test, y_pred)
#     recallList.append(recall)
#
#     print(f"\n***K-fold: {foldCount}")
#     foldCount += 1
#     print('Accuracy: ', accuracy)
#     print('Precision: {0:0.2f}'.format(precision))
#     print('Recall: {0:0.2f}'.format(recall))
#
# # Calculate and display overall accuracy and standard deviation
# print("\nAccuracy and Standard Deviation For All Folds:")
# print("*********************************************")
# print("Average accuracy: " + str(np.mean(accuracyList)))
# print("Accuracy std: " + str(np.std(accuracyList)))
# print("Average precision: " + str(np.mean(precisionList)))
# print("Precision std: " + str(np.std(precisionList)))
# print("Average recall: " + str(np.mean(recallList)))
# print("Recall std: " + str(np.std(recallList)))



# Example 9 linear regression without scalling

# import pandas as pd
# import numpy as np
# from sklearn import datasets
# from   sklearn.model_selection import train_test_split
# import statsmodels.api         as sm
# import numpy                   as np
# from   sklearn                 import metrics
#
# wine = datasets.load_wine()
# dataset = pd.DataFrame(
#     data=np.c_[wine['data'], wine['target']],
#     columns=wine['feature_names'] + ['target']
# )
#
# # Create copy to prevent overwrite.
# X = dataset.copy()
# del X['target']         # Remove target variable
# del X['hue']            # Remove unwanted features
# del X['ash']
# del X['magnesium']
# del X['malic_acid']
# del X['alcohol']
#
# y = dataset['target']
#
# # Adding an intercept *** This is requried ***. Don't forget this step.
# X = sm.add_constant(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# from sklearn.preprocessing import RobustScaler
# sc_x     = RobustScaler()
# X_train_scaled = sc_x.fit_transform(X_train)
# X_test_scaled  = sc_x.transform(X_test)
#
# # Create y scaler. Only scale y_train since evaluation
# # will use the actual size y_test.
# sc_y           = RobustScaler()
# y_train_scaled = sc_y.fit_transform(np.array(y_train).reshape(-1,1))
#
# model       = sm.OLS(y_train_scaled, X_train_scaled).fit()
# unscaledPredictions = model.predict(X_test_scaled) # make predictions
#
# # Rescale predictions back to actual size range.
# predictions = sc_y.inverse_transform(np.array(unscaledPredictions).reshape(-1,1))
#
#
# print(model.summary())
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



# Example 11 saving and reusing scalers

import pandas as pd
import numpy as np
from sklearn import datasets
from   sklearn.model_selection import train_test_split
import statsmodels.api         as sm
import numpy                   as np
from   sklearn                 import metrics

wine = datasets.load_wine()
dataset = pd.DataFrame(
    data=np.c_[wine['data'], wine['target']],
    columns=wine['feature_names'] + ['target']
)

# Create copy to prevent overwrite.
X = dataset.copy()
del X['target']         # Remove target variable
del X['hue']            # Remove unwanted features
del X['ash']
del X['magnesium']
del X['malic_acid']
del X['alcohol']

y = dataset['target']

# Adding an intercept *** This is requried ***. Don't forget this step.
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import RobustScaler
sc_x     = RobustScaler()
X_train_scaled = sc_x.fit_transform(X_train)

# Create y scaler. Only scale y_train since evaluation
# will use the actual size y_test.
sc_y           = RobustScaler()
y_train_scaled = sc_y.fit_transform(np.array(y_train).reshape(-1,1))

# Save the fitted scalers.
from pickle import dump, load
dump(sc_x, open('../COMP3948Labs/sc_x.pkl', 'wb'))
dump(sc_y, open('../COMP3948Labs/sc_y.pkl', 'wb'))

# Build model with training data.
model       = sm.OLS(y_train_scaled, X_train_scaled).fit()

# Save the trained model
dump(model, open('../COMP3948Labs/model.pkl', 'wb'))

# Load the scalers.
loaded_scalerX = load(open('../COMP3948Labs/sc_x.pkl', 'rb'))
loaded_scalery = load(open('../COMP3948Labs/sc_y.pkl', 'rb'))

# Load the trained model
loaded_model = load(open('../COMP3948Labs/model.pkl', 'rb'))

X_test_scaled = loaded_scalerX.transform(X_test)
unscaledPredictions = model.predict(X_test_scaled) # make predictions

# Rescale predictions back to actual size range.
predictions = loaded_scalery.inverse_transform(
               np.array(unscaledPredictions).reshape(-1,1))

print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))
