# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model    import LinearRegression
# from sklearn                 import metrics
# import statsmodels.api       as sm
# import numpy                 as np
#
# PATH     = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "USA_Housing.csv"
# dataset  = pd.read_csv(PATH + CSV_DATA,
#                       skiprows=1,
#                       encoding = "ISO-8859-1", sep=',',
#                       names=('Avg. Area Income','Avg. Area House Age',
#                              'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms',
#                              "Area Population", 'Price', "Address"))
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# print("FIRST THREE ROWS OF DATASET \n", dataset.head(3), "\n")
#
# # Statistical summary
# print("STATISTICAL SUMMARY \n", dataset.describe(), "\n")
#
# # # Compute the correlation matrix
# # numeric_dataset = dataset.select_dtypes(include=['float64', 'int64'])
# # # Compute the correlation matrix for numeric columns
# # corr = numeric_dataset.corr()
# # # Display correlation matrix
# # print("CORRELATION MATRIX\n", corr, "\n")
# # # Heatmap of the correlation matrix for numeric columns
# # sns.set(rc={'figure.figsize': (8, 6)})
# # sns.heatmap(corr, annot=True, linewidths=0.1, vmin=-1, vmax=1, cmap="YlGnBu")
# # plt.tight_layout()
# # plt.show()
#
# X = dataset[['Avg. Area Income','Avg. Area House Age',
#              'Avg. Area Number of Rooms', 'Area Population']]
#
# # Adding an intercept *** This is required ***. Don't forget this step.
# # * This step is only needed when using sm.OLS. *
# # The intercept centers the error residuals around zero
# # which helps to avoid over-fitting.
# X = sm.add_constant(X)
#
# y = dataset['Price']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# model = sm.OLS(y_train, X_train).fit()
# predictions = model.predict(X_test) # make the predictions by the model
#
# print(model.summary())
#
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#
#
# def plotPredictionVsActual(plt, title, y_test, predictions):
#     plt.scatter(y_test, predictions)
#     plt.xlabel("Actual")
#     plt.ylabel("Predicted")
#     plt.title('Predicted (Y) vs. Actual (X): ' + title)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
#     plt.legend(['Prediction Line', 'Predicted vs Actual'])
#
#
# def plotResidualsVsActual(plt, title, y_test, predictions):
#     residuals = y_test - predictions
#     plt.scatter(y_test, residuals, label='Residuals vs Actual')
#     plt.xlabel("Actual")
#     plt.ylabel("Residual")
#     plt.title('Error Residuals (Y) vs. Actual (X): ' + title)
#     plt.axhline(0, color='k', linestyle='--')
#
#
# def plotResidualHistogram(plt, title, y_test, predictions, bins):
#     residuals = y_test - predictions
#     plt.hist(residuals, bins=bins, edgecolor='black')
#     plt.xlabel("Residual")
#     plt.ylabel("Frequency")
#     plt.title('Error Residual Frequency: ' + title)
#
#
# def drawValidationPlots(title, bins, y_test, predictions):
#     plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
#
#     plt.subplot(1, 3, 1)
#     plotPredictionVsActual(plt, title, y_test, predictions)
#
#     plt.subplot(1, 3, 2)
#     plotResidualsVsActual(plt, title, y_test, predictions)
#
#     plt.subplot(1, 3, 3)
#     plotResidualHistogram(plt, title, y_test, predictions, bins)
#
#     plt.tight_layout()
#     plt.show()
#
# BINS = 8
# TITLE = "Model B - Housing Price Prediction"
#
# drawValidationPlots(TITLE, BINS, y_test, predictions)