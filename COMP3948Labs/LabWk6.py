# import pandas  as pd
# import numpy   as np
#
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
#
# FOLDER_PATH =  '/Users/dandidac/Documents/COMP 3948/Data Sets/'
# FILE        = 'employee_turnover.csv'
# df          = pd.read_csv(FOLDER_PATH + FILE)
# print(df)
#
# # Separate into x and y values.
# predictorVariables = list(df.keys())
# predictorVariables.remove('turnover')
# print(predictorVariables)
#
# # Create X and y values.
# X = df[predictorVariables]
# y = df['turnover']
#
# # Import the necessary libraries first
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# # You imported the libraries to run the experiments. Now, let's see it in action.
#
# # Show chi-square scores for each feature.
# # There is 1 degree freedom since 1 predictor during feature evaluation.
# # Generally, >=3.8 is good)
# test      = SelectKBest(score_func=chi2, k=3)
# chiScores = test.fit(X, y) # Summarize scores
# np.set_printoptions(precision=3)
#
# print("\nPredictor variables: " + str(predictorVariables))
# print("Predictor Chi-Square Scores: " + str(chiScores.scores_))
#
# # Another technique for showing the most statistically
# # significant variables involves the get_support() function.
# cols = chiScores.get_support(indices=True)
# print(cols)
# features = X.columns[cols]
# print(np.array(features))
#
# from   sklearn.model_selection import train_test_split
# from   sklearn.linear_model    import LogisticRegression
#
# # Re-assign X with significant columns only after chi-square test.
# X = df[['experience', 'age', 'way', 'industry']]
#
# # Split data.
# X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25,
#                                                  random_state=0)
#
# # Build logistic regression model and make predictions.
# logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',
#                                    random_state=0)
# logisticModel.fit(X_train,y_train)
# y_pred=logisticModel.predict(X_test)
# print(y_pred)
#
# # Show confusion matrix and accuracy scores.
# from   sklearn                 import metrics
# cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix")
# print(cm)
#
# TN = cm[0][0] # = 3 True Negative  (Col 0, Row 0)
# FN = cm[0][1] # = 0 False Negative (Col 0, Row 1)
# FP = cm[1][0] # = 2 False Positive (Col 1, Row 0)
# TP = cm[1][1] # = 5 True Positive  (Col 1, Row 1)
#
# print("")
# print("True Negative:  " + str(TN))
# print("False Negative: " + str(FN))
# print("False Positive: " + str(FP))
# print("True Positive:  " + str(TP))
#
# precision = (TP/(FP + TP))
# print("\nPrecision:  " + str(round(precision, 3)))
#
# recall = (TP/(TP + FN))
# print("Recall:     " + str(round(recall,3)))
#
# F1 = 2*((precision*recall)/(precision+recall))
# print("F1:         " + str(round(F1,3)))





# # Load libraries
# from sklearn.linear_model    import LogisticRegression
# from sklearn                 import datasets
# from sklearn.model_selection import train_test_split
# import pandas as pd
#
# # Load data from sklearn library.
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# # Split data.
# X_train,X_test,y_train,y_test = train_test_split(
#         X, y, test_size=0.25,random_state=0)
#
# # Create one-vs-rest logistic regression object
# clf = LogisticRegression(
#     random_state=0,
#     multi_class='multinomial', solver='newton-cg')
#
# # Train model
# model  = clf.fit(X_train, y_train)
#
# # Predict class
# y_pred = model.predict(X_test)
# print(y_pred)
#
# # View predicted probabilities
# y_prob = model.predict_proba(X_test)
# print(y_prob)
#
# # Show precision, recall and F1 scores for all classes.
# from   sklearn                 import metrics
# precision = metrics.precision_score(y_test, y_pred, average=None)
# recall    = metrics.recall_score(   y_test, y_pred, average=None)
# f1        = metrics.f1_score(       y_test, y_pred, average=None)
#
# print("Precision: " + str(precision))
# print("Recall: "    + str(recall))
# print("F1: "        + str(f1))





# Load libraries
from sklearn.linear_model    import LogisticRegression
from sklearn                 import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

PATH = '/Users/dandidac/Documents/COMP 3948/Data Sets/'
FILE = 'glass.csv'
df   = pd.read_csv(PATH + FILE)

# Show DataFrame contents.
print(df.head())

# Get X values and remove target column.
# Make copy to avoid over-writing.
X = df.copy()
del X['Type']

# Get y values
y = df['Type']

# Split data.
X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.25,random_state=0)

# Create one-vs-rest logistic regression object
clf = LogisticRegression(
    random_state=0,
    multi_class='multinomial', solver='newton-cg')

# Train model
model  = clf.fit(X_train, y_train)

# Predict class
y_pred = model.predict(X_test)
print(y_pred)

# View predicted probabilities
y_prob = model.predict_proba(X_test)
print(y_prob)

# Show precision, recall and F1 scores for all classes.
from   sklearn                 import metrics
precision = metrics.precision_score(y_test, y_pred, average=None)
recall    = metrics.recall_score(   y_test, y_pred, average=None)
f1        = metrics.f1_score(       y_test, y_pred, average=None)

print("Precision: " + str(precision))
print("Recall: "    + str(recall))
print("F1: "        + str(f1))
print(df["Type"])
# Get unique class labels
# classes = df['Type'].unique()
#
# # Print scores with corresponding class
# for i, class_label in enumerate(classes):
#     print(f"Class {class_label}:")
#     print(f"  Precision: {precision[i]}")
#     print(f"  Recall:    {recall[i]}")
#     print(f"  F1-Score:  {f1[i]}")






# Show confusion matrix and accuracy scores.

# import pandas  as pd
# import numpy   as np
#
# # Setup data.
# candidates = {'gmat': [780,750,690,710,680,730,690,720,
#  740,690,610,690,710,680,770,610,580,650,540,590,620,
#  600,550,550,570,670,660,580,650,660,640,620,660,660,
#  680,650,670,580,590,690],
#               'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,
#  3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,
#  3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,
#  3.3,3.3,2.3,2.7,3.3,1.7,3.7],
#               'work_experience': [3,4,3,5,4,6,1,4,5,
#  1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,
#  5,1,2,1,4,5],
#               'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,
#  1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,
#  0,0,1]}
#
# df = pd.DataFrame(candidates,columns= ['gmat', 'gpa',
#                                        'work_experience','admitted'])
# print(df)
#
# # Separate into x and y values.
# predictorVariables = ['gmat', 'gpa','work_experience']
# X = df[predictorVariables]
# y = df['admitted']
#
# # Import the necessary libraries first
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# # You imported the libraries to run the experiments. Now, let's see it in action.
#
# # Show chi-square scores for each feature.
# # There is 1 degree freedom since 1 predictor during feature evaluation.
# # Generally, >=3.8 is good)
# test      = SelectKBest(score_func=chi2, k=3)
# chiScores = test.fit(X, y) # Summarize scores
# np.set_printoptions(precision=3)
#
# print("\nPredictor variables: " + str(predictorVariables))
# print("Predictor Chi-Square Scores: " + str(chiScores.scores_))
#
# # Another technique for showing the most statistically
# # significant variables involves the get_support() function.
# cols = chiScores.get_support(indices=True)
# print(cols)
# features = X.columns[cols]
# print(np.array(features))
#
# from   sklearn.model_selection import train_test_split
# from   sklearn.linear_model    import LogisticRegression
#
# # Re-assign X with significant columns only after chi-square test.
# X = df[['gmat', 'work_experience']]
#
# # Split data.
# X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25,
#                                                  random_state=0)
#
# # Build logistic regression model and make predictions.
# logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',
#                                    random_state=0)
# logisticModel.fit(X_train,y_train)
# y_pred=logisticModel.predict(X_test)
# print(y_pred)
#
# from   sklearn                 import metrics
# cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix")
# print(cm)
#
#
# TN = cm[0][0] # = 3 True Negative  (Col 0, Row 0)
# FN = cm[0][1] # = 0 False Negative (Col 0, Row 1)
# FP = cm[1][0] # = 2 False Positive (Col 1, Row 0)
# TP = cm[1][1] # = 5 True Positive  (Col 1, Row 1)
#
# print("")
# print("True Negative:  " + str(TN))
# print("False Negative: " + str(FN))
# print("False Positive: " + str(FP))
# print("True Positive:  " + str(TP))
#
# precision = (TP/(FP + TP))
# print("\nPrecision:  " + str(round(precision, 3)))
#
# recall = (TP/(TP + FN))
# print("Recall:     " + str(round(recall,3)))
#
# F1 = 2*((precision*recall)/(precision+recall))
# print("F1:         " + str(round(F1,3)))
