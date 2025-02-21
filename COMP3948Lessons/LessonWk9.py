# example 2
# Importing required libraries:
# import warnings
# warnings.simplefilter(action="ignore", category=FutureWarning)
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from   sklearn  import metrics
# from   sklearn.model_selection import train_test_split
# from   sklearn.linear_model    import LogisticRegression
# from sklearn.feature_selection import RFE
#
# # Read the data:
# PATH = '/Users/dandidac/Documents/COMP 3948/Data Sets/'
# df = pd.read_csv(PATH + "Divorce.csv", header = 0)
#
# # Seperate the target and independent variable
# X = df.copy()       # Create separate copy to prevent unwanted tampering of data.
# del X['Divorce']     # Delete target variable.
#
# # Target variable
# y = df['Divorce']
#
# # Create the object of the model
# model = LogisticRegression()
#
# # Specify the number of  features to select
# rfe = RFE(model, n_features_to_select = 8)
#
# # fit the model
# rfe = rfe.fit(X, y)
#
# # Please uncomment the following lines to see the result
# print('\n\nFEATURES SELECTED\n\n')
# print(rfe.support_)
#
# # Show top features.
# for i in range(0, len(X.keys())):
#     if(rfe.support_[i]):
#         print(X.keys()[i])
#
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# def buildAndEvaluateClassifier(features, X, y):
#     # Re-assign X with significant columns only after chi-square test.
#     X = X[features]
#
#     # Split data.
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#
#     # Perform logistic regression.
#     logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear', random_state=0)
#
#     # Fit the model.
#     logisticModel.fit(X_train, y_train)
#     y_pred = logisticModel.predict(X_test)
#     # print(y_pred)
#
#     # Show accuracy scores.
#     print('Results without scaling:')
#
#     # Show confusion matrix
#     cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
#     print("\nConfusion Matrix")
#     print(cm)
#
#     print("Recall:    " + str(recall_score(y_test, y_pred)))
#     print("Precision: " + str(precision_score(y_test, y_pred)))
#     print("F1:        " + str(f1_score(y_test, y_pred)))
#     print("Accuracy:  " + str(accuracy_score(y_test, y_pred)))
# features = ['Q3', 'Q6', 'Q17', 'Q18', 'Q26', 'Q39', 'Q40', 'Q49']
# buildAndEvaluateClassifier(features, X, y)
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
# Exercise 4

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot       as plt
# import pandas                  as pd
# import numpy                   as np
#
# PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "bank-additional-full.csv"
# df = pd.read_csv(PATH + CSV_DATA,
#                  skiprows=1,  # Don't include header row as part of data.
#                  encoding="ISO-8859-1", sep=';',
#                  names=(
# "age", "job", "marital", "education", "default", "housing", "loan", "contact",
# "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
# "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"))
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
#
# print(df.head())
# print(df.describe().transpose())
# print(df.info())
#
# targetList = []
# for i in range(0, len(df)):
#     if (df.loc[i]['y'] == 'yes'):
#         targetList.append(1)
#     else:
#         targetList.append(0)
# df['target'] = targetList
#
# tempDf = df[["job", "marital", "education", "default","housing", "loan", "contact", "month", "day_of_week", "poutcome"]]  # Isolate columns
# dummyDf = pd.get_dummies(tempDf, columns=["job", "marital", "education", "default",
# "housing", "loan", "contact", "month", "day_of_week", "poutcome"])  # Get dummies
# df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original df
#
# X = df[[
# "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
#     "cons.conf.idx", "euribor3m", "nr.employed", "job_admin.", "job_blue-collar",
#     "job_entrepreneur", "job_housemaid", "job_management", "job_retired",
#     "job_self-employed", "job_services", "job_student", "job_technician", "job_unemployed",
#     "job_unknown", "marital_divorced", "marital_married", "marital_single",
#     "marital_unknown", "education_basic.4y", "education_basic.6y", "education_basic.9y",
#     "education_high.school", "education_illiterate", "education_professional.course",
#     "education_university.degree", "education_unknown", "default_no",
#     "default_unknown", "default_yes", "housing_no", "housing_unknown", "housing_yes",
#     "loan_no", "loan_unknown", "loan_yes", "contact_cellular", "contact_telephone",
#     "month_apr", "month_aug", "month_dec", "month_jul", "month_jun", "month_mar",
#     "month_may", "month_nov", "month_oct", "month_sep", "day_of_week_fri",
#     "day_of_week_mon", "day_of_week_thu", "day_of_week_tue", "day_of_week_wed",
#     "poutcome_failure", "poutcome_nonexistent", "poutcome_success", ]]
# y = df[['target']].values.ravel()
# print(X.head())
#
# # Create the object of the model
# model = LogisticRegression(solver='liblinear')
#
# # Specify the number of  features to select
# rfe = RFE(estimator=model, n_features_to_select = 15)
#
# # fit the model
# rfe = rfe.fit(X, y)
#
# # Please uncomment the following lines to see the result
# print('\n\nFEATURES SELECTED\n\n')
# print(rfe.support_)
#
# # Show top features.
# for i in range(0, len(X.keys())):
#     if(rfe.support_[i]):
#         print(X.keys()[i])



# Exercise 5    Chi

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import roc_auc_score, confusion_matrix
# import matplotlib.pyplot       as plt
# import pandas                  as pd
# import numpy                   as np
#
# PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "bank-additional-full.csv"
# df = pd.read_csv(PATH + CSV_DATA,
#                  skiprows=1,
#                  encoding="ISO-8859-1", sep=';',
#                  names=(
# "age", "job", "marital", "education", "default", "housing", "loan", "contact",
# "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
# "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"))
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
#
# print(df.head())
# print(df.describe().transpose())
# print(df.info())
#
# targetList = []
# for i in range(0, len(df)):
#     if (df.loc[i]['y'] == 'yes'):
#         targetList.append(1)
#     else:
#         targetList.append(0)
# df['target'] = targetList
#
# tempDf = df[["job", "marital", "education", "default","housing", "loan", "contact", "month", "day_of_week", "poutcome"]]  # Isolate columns
# dummyDf = pd.get_dummies(tempDf, columns=["job", "marital", "education", "default",
# "housing", "loan", "contact", "month", "day_of_week", "poutcome"])  # Get dummies
# df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original df
#
# X = df[[
# "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
#     "cons.conf.idx", "euribor3m", "nr.employed", "job_admin.", "job_blue-collar",
#     "job_entrepreneur", "job_housemaid", "job_management", "job_retired",
#     "job_self-employed", "job_services", "job_student", "job_technician", "job_unemployed",
#     "job_unknown", "marital_divorced", "marital_married", "marital_single",
#     "marital_unknown", "education_basic.4y", "education_basic.6y", "education_basic.9y",
#     "education_high.school", "education_illiterate", "education_professional.course",
#     "education_university.degree", "education_unknown", "default_no",
#     "default_unknown", "default_yes", "housing_no", "housing_unknown", "housing_yes",
#     "loan_no", "loan_unknown", "loan_yes", "contact_cellular", "contact_telephone",
#     "month_apr", "month_aug", "month_dec", "month_jul", "month_jun", "month_mar",
#     "month_may", "month_nov", "month_oct", "month_sep", "day_of_week_fri",
#     "day_of_week_mon", "day_of_week_thu", "day_of_week_tue", "day_of_week_wed",
#     "poutcome_failure", "poutcome_nonexistent", "poutcome_success", ]]
# y = df[['target']].values.ravel()
# print(X.head())
#
# from sklearn.linear_model import LogisticRegression
# # Show chi-square scores for each feature.
# # There is 1 degree freedom since 1 predictor during feature evaluation.
# # Generally, >=3.8 is good)
# from sklearn.feature_selection import chi2
# from sklearn.feature_selection import SelectKBest
# test      = SelectKBest(score_func=chi2, k=15)
#
# # Use scaled data to fit KBest
# XScaled   = MinMaxScaler().fit_transform(X)
# chiScores = test.fit(XScaled, y) # Summarize scores
# np.set_printoptions(precision=3)
#
# # Search here for insignificant features.
# print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))
#
# # Create a sorted list of the top features.
# dfFeatures = pd.DataFrame()
# for i in range(0, len(chiScores.scores_)):
#     headers      = list(X.keys())
#     featureObject = {"feature":headers[i], "chi-square score":chiScores.scores_[i]}
#     dfFeatures    = dfFeatures._append(featureObject, ignore_index=True)
#
# print("\nTop Features")
# dfFeatures = dfFeatures.sort_values(by=['chi-square score'], ascending=False)
# print(dfFeatures.head(15))
#
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# def buildAndEvaluateClassifier(features, X, y):
#     # Re-assign X with significant columns only after chi-square test.
#     X = X[features]
#     print("Features used in logistic regression:", X.columns)
#     print("Number of features:", X.shape[1])
#     # Split data.
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#
#     # Perform logistic regression.
#     logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear')
#
#     # Fit the model.
#     logisticModel.fit(X_train, y_train)
#     y_pred = logisticModel.predict(X_test)
#     y_prob = logisticModel.predict_proba(X_test)
#     # print(y_pred)
#
#     # Show accuracy scores.
#     print('Results without scaling:')
#
#     print("Recall:    " + str(recall_score(y_test, y_pred)))
#     print("Precision: " + str(precision_score(y_test, y_pred)))
#     print("F1:        " + str(f1_score(y_test, y_pred)))
#     print("Accuracy:  " + str(accuracy_score(y_test, y_pred)))
#     return X_test, y_test, y_pred, y_prob
# top_features = dfFeatures['feature'].head(15)
# X_test, y_test, y_pred, y_prob =\
# buildAndEvaluateClassifier(top_features, X, y)
#
# from sklearn.metrics           import roc_curve
# from sklearn.metrics           import roc_auc_score
#
# auc = roc_auc_score(y_test, y_prob[:, 1],)
# print('Logistic: ROC AUC=%.3f' % (auc))
#
# # calculate roc curves
# lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob[:, 1])
# plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# plt.plot([0,1], [0,1], '--', label='No Skill')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()
#
# print(df['target'].value_counts())
#
# cm = confusion_matrix(y_test, y_pred)
# print(cm)



# Exercise 6    RFE

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, roc_curve, \
#     confusion_matrix
# import matplotlib.pyplot       as plt
# import pandas                  as pd
# import numpy                   as np
#
# PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "bank-additional-full.csv"
# df = pd.read_csv(PATH + CSV_DATA,
#                  skiprows=1,  # Don't include header row as part of data.
#                  encoding="ISO-8859-1", sep=';',
#                  names=(
# "age", "job", "marital", "education", "default", "housing", "loan", "contact",
# "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
# "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"))
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
#
# print(df.head())
# print(df.describe().transpose())
# print(df.info())
#
# targetList = []
# for i in range(0, len(df)):
#     if (df.loc[i]['y'] == 'yes'):
#         targetList.append(1)
#     else:
#         targetList.append(0)
# df['target'] = targetList
#
# tempDf = df[["job", "marital", "education", "default","housing", "loan", "contact", "month", "day_of_week", "poutcome"]]  # Isolate columns
# dummyDf = pd.get_dummies(tempDf, columns=["job", "marital", "education", "default",
# "housing", "loan", "contact", "month", "day_of_week", "poutcome"])  # Get dummies
# df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original df
#
# X = df[[
# "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
#     "cons.conf.idx", "euribor3m", "nr.employed", "job_admin.", "job_blue-collar",
#     "job_entrepreneur", "job_housemaid", "job_management", "job_retired",
#     "job_self-employed", "job_services", "job_student", "job_technician", "job_unemployed",
#     "job_unknown", "marital_divorced", "marital_married", "marital_single",
#     "marital_unknown", "education_basic.4y", "education_basic.6y", "education_basic.9y",
#     "education_high.school", "education_illiterate", "education_professional.course",
#     "education_university.degree", "education_unknown", "default_no",
#     "default_unknown", "default_yes", "housing_no", "housing_unknown", "housing_yes",
#     "loan_no", "loan_unknown", "loan_yes", "contact_cellular", "contact_telephone",
#     "month_apr", "month_aug", "month_dec", "month_jul", "month_jun", "month_mar",
#     "month_may", "month_nov", "month_oct", "month_sep", "day_of_week_fri",
#     "day_of_week_mon", "day_of_week_thu", "day_of_week_tue", "day_of_week_wed",
#     "poutcome_failure", "poutcome_nonexistent", "poutcome_success", ]]
# y = df[['target']].values.ravel()
# print(X.head())
#
# # Create the object of the model
# model = LogisticRegression(solver='liblinear')
#
# # Specify the number of  features to select
# rfe = RFE(estimator=model, n_features_to_select = 15)
#
# # fit the model
# rfe = rfe.fit(X, y)
#
# # Please uncomment the following lines to see the result
# print('\n\nFEATURES SELECTED\n\n')
# print(rfe.support_)
#
# # Show top features.
# for i in range(0, len(X.keys())):
#     if(rfe.support_[i]):
#         print(X.keys()[i])
#
# # Function to build and evaluate classifier
# def build_and_evaluate_classifier(X, y):
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#
#     # Perform logistic regression
#     logistic_model = LogisticRegression(fit_intercept=True, solver='liblinear')
#
#     # Fit the model
#     logistic_model.fit(X_train, y_train)
#     y_pred = logistic_model.predict(X_test)
#     y_prob = logistic_model.predict_proba(X_test)[:, 1]
#
#     # Show evaluation metrics
#     print("Recall:", recall_score(y_test, y_pred))
#     print("Precision:", precision_score(y_test, y_pred))
#     print("F1 Score:", f1_score(y_test, y_pred))
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     return y_test, y_pred, y_prob
#
# # Evaluate classifier with selected features
# y_test, y_pred, y_prob = build_and_evaluate_classifier(X, y)
#
# # ROC AUC
# auc = roc_auc_score(y_test, y_prob)
# print("Logistic: ROC AUC = %.3f" % auc)
#
# # Plot ROC curve
# lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob)
# plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()
#
# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", cm)
#
# # Display target value counts
# print(df['target'].value_counts())



# Exercise 8    forward feature

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, accuracy_score, f1_score, precision_score, \
    recall_score
import matplotlib.pyplot       as plt
import pandas                  as pd
import numpy                   as np

PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
CSV_DATA = "bank-additional-full.csv"
df = pd.read_csv(PATH + CSV_DATA,
                 skiprows=1,  # Don't include header row as part of data.
                 encoding="ISO-8859-1", sep=';',
                 names=(
"age", "job", "marital", "education", "default", "housing", "loan", "contact",
"month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
"emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"))
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df.head())
print(df.describe().transpose())
print(df.info())

targetList = []
for i in range(0, len(df)):
    if (df.loc[i]['y'] == 'yes'):
        targetList.append(1)
    else:
        targetList.append(0)
df['target'] = targetList

tempDf = df[["job", "marital", "education", "default","housing", "loan", "contact", "month", "day_of_week", "poutcome"]]  # Isolate columns
dummyDf = pd.get_dummies(tempDf, columns=["job", "marital", "education", "default",
"housing", "loan", "contact", "month", "day_of_week", "poutcome"])  # Get dummies
df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original df

X = df[[
"age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "job_admin.", "job_blue-collar",
    "job_entrepreneur", "job_housemaid", "job_management", "job_retired",
    "job_self-employed", "job_services", "job_student", "job_technician", "job_unemployed",
    "job_unknown", "marital_divorced", "marital_married", "marital_single",
    "marital_unknown", "education_basic.4y", "education_basic.6y", "education_basic.9y",
    "education_high.school", "education_illiterate", "education_professional.course",
    "education_university.degree", "education_unknown", "default_no",
    "default_unknown", "default_yes", "housing_no", "housing_unknown", "housing_yes",
    "loan_no", "loan_unknown", "loan_yes", "contact_cellular", "contact_telephone",
    "month_apr", "month_aug", "month_dec", "month_jul", "month_jun", "month_mar",
    "month_may", "month_nov", "month_oct", "month_sep", "day_of_week_fri",
    "day_of_week_mon", "day_of_week_thu", "day_of_week_tue", "day_of_week_wed",
    "poutcome_failure", "poutcome_nonexistent", "poutcome_success", ]]
y = df[['target']].values.ravel()
print(X.head())

ffs = f_regression(X, y)

selected_features = [X.columns[i] for i, f in enumerate(ffs[0]) if f >= 700]
X_selected = X[selected_features]
print(f"Number of selected features: {len(selected_features)}")
print("Selected features:")
for feature in selected_features:
    print(feature)
# Function to build and evaluate classifier
def build_and_evaluate_classifier(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Fit the model
    logistic_model = LogisticRegression(solver='liblinear')
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    y_prob = logistic_model.predict_proba(X_test)[:, 1]

    # Show evaluation metrics
    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return y_test, y_pred, y_prob

# Evaluate classifier with selected features
y_test, y_pred, y_prob = build_and_evaluate_classifier(X_selected, y)

# ROC AUC
auc = roc_auc_score(y_test, y_prob)
print("Logistic: ROC AUC = %.3f" % auc)

# Plot ROC curve
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Display target value counts
print(df['target'].value_counts())