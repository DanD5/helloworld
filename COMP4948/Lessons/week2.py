import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

# File path and dataset
PATH = "/Users/dandidac/Documents/Data Sets/"
CSV_DATA = "petrol_consumption.csv"
dataset = pd.read_csv(PATH + CSV_DATA)

# Features and target
X = dataset[['Petrol_tax', 'Average_income', 'Population_Driver_licence(%)']]
y = dataset['Petrol_Consumption'].values

# Adding an intercept for OLS regression
X_withConst = sm.add_constant(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_withConst, y, test_size=0.2, random_state=0)

# Linear regression with OLS
def performLinearRegression(X_train, X_test, y_train, y_test):
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    return predictions

predictions_ols = performLinearRegression(X_train, X_test, y_train, y_test)

# Scaling the features and target for SGD
scalerX = StandardScaler()
scalerY = StandardScaler()

X_train_scaled = scalerX.fit_transform(X_train)
X_test_scaled = scalerX.transform(X_test)

# Reshape y to fit scaler requirements
y_train_reshaped = y_train.reshape(-1, 1)
y_train_scaled = scalerY.fit_transform(y_train_reshaped)

from sklearn.linear_model import ElasticNet

bestRMSE = 100000.03
def performElasticNetRegression(X_train, X_test, y_train, y_test, alpha, l1ratio, bestRMSE,
                                bestAlpha, bestL1Ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1ratio)
    # fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n***ElasticNet Regression Coefficients ** alpha=" + str(alpha)
          + " l1ratio=" + str(l1ratio))
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(model.intercept_)
    print(model.coef_)
    try:
        if(rmse < bestRMSE):
            bestRMSE = rmse
            bestAlpha = alpha
            bestL1Ratio = l1ratio
        print('Root Mean Squared Error:', rmse)
    except:
        print("rmse =" + str(rmse))

    return bestRMSE, bestAlpha, bestL1Ratio

alphaValues = [0, 0.00001, 0.0001, 0.001, 0.01, 0.18]
l1ratioValues = [0, 0.25, 0.5, 0.75, 1]
bestAlpha   = 0
bestL1Ratio = 0

for i in range(0, len(alphaValues)):
    for j in range(0, len(l1ratioValues)):
        bestRMSE, bestAlpha, bestL1Ratio = performElasticNetRegression(
                         X_train, X_test, y_train, y_test,
                         alphaValues[i], l1ratioValues[j], bestRMSE,
                         bestAlpha, bestL1Ratio)

print("Best RMSE " + str(bestRMSE) + " Best alpha: " + str(bestAlpha)
      + "  " + "Best l1 ratio: " + str(bestL1Ratio))
