# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# # Create DataFrame.
# dataSet = {'days': [0.2, 0.32, 0.38, 0.41, 0.43],
#            'growth': [0.1, 0.15, 0.4, 0.6, 0.44]}
# df = pd.DataFrame(dataSet, columns=['days', 'growth'])
#
# # Store x and y values.
# X = df['days']
# target = df['growth']
#
# # Create training set with 80% of data and test set with 20% of data.
# X_train, X_test, y_train, y_test = train_test_split(
#     X, target, train_size=0.8
# )
#
# # Combine the data into a DataFrame for custom formatting
# output_df = pd.DataFrame({
#     'X_train': X_train.reset_index(drop=True),
#     'y_train': y_train.reset_index(drop=True),
#     'X_test': X_test.reset_index(drop=True),
#     'y_test': y_test.reset_index(drop=True)
# })
#
# # Print in the specific format requested
# print(output_df.to_string(index=False, header=True, justify='right'))

import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from sklearn import metrics
import math

def performSimpleRegression():
    # Initialize collection of X & Y pairs like those used in example 5.
    data = [[0.2, 0.1], [0.32, 0.15], [0.38, 0.4], [0.41, 0.6], [0.43, 0.44]]

    # Create data frame.
    dfSample = pd.DataFrame(data, columns=['X', 'target'])

    # Create training set with 80% of data and test set with 20% of data.
    X_train, X_test, y_train, y_test = train_test_split(
        dfSample['X'], dfSample['target'], train_size=0.8
    )

    # Create DataFrame with test data.
    dataTrain = {"X": X_train, "target": y_train}
    dfTrain = pd.DataFrame(dataTrain, columns=['X', 'target'])

    # Generate model to predict target using X.
    model = ols('target ~ X', data=dfTrain).fit()
    y_prediction = model.predict(X_test)

    # Present X_test, y_test, y_predict and error sum of squares.
    data = {"X_test": X_test, "y_test": y_test, "y_prediction": y_prediction}
    dfResult = pd.DataFrame(data, columns=['X_test', 'y_test', 'y_prediction'])
    dfResult['y_test - y_pred'] = (dfResult['y_test'] - dfResult['y_prediction'])
    dfResult['(y_test - y_pred)^2'] = (dfResult['y_test'] - dfResult['y_prediction']) ** 2

    # Present X_test, y_test, y_predict and error sum of squares.
    print(dfResult)

    # Manually calculate the deviation between actual and predicted values.
    rmse = math.sqrt(dfResult['(y_test - y_pred)^2'].sum() / len(dfResult))
    print("RMSE is average deviation between actual and predicted values: "
          + str(rmse))

    # Show faster way to calculate deviation between actual and predicted values.
    rmse2 = math.sqrt(metrics.mean_squared_error(y_test, y_prediction))
    print("The automated root mean square error calculation is: " + str(rmse2))


performSimpleRegression()

