# import matplotlib.pyplot    as plt
# from sklearn.metrics import mean_squared_error
# import pandas               as pd
# import statsmodels.api      as sm
# import numpy                as np
#
# #------------------------------------------------
# # Shows plot of x vs. y.
# #------------------------------------------------
# def showXandYplot(x,y, xtitle, title):
#     plt.figure(figsize=(8, 4))
#     plt.plot(x,y,color='blue')
#     plt.title(title)
#     plt.xlabel(xtitle)
#     plt.ylabel('y')
#     plt.show()
#
# #------------------------------------------------
# # Shows plot of actual vs. predicted and RMSE.
# #------------------------------------------------
# def showResidualPlotAndRMSE(x, y, predictions):
#     xmax      = max(x)
#     xmin      = min(x)
#     residuals = y - predictions
#
#     plt.figure(figsize=(8, 3))
#     plt.title('x and y')
#     plt.plot([xmin,xmax],[0,0],'--',color='black')
#     plt.title("Residuals")
#     plt.scatter(x,residuals,color='red')
#     plt.show()
#
#     # Calculate RMSE
#     mse = mean_squared_error(y,predictions)
#     rmse = np.sqrt(mse)
#     print("RMSE: " + str(rmse))
#
# # Section A: Define the raw data.
# x = [0.01, 0.2, 0.5, 0.7, 0.9,1,2,3,4,5,6,10,11,12,13,14,15,16,17,18,19,20]
# y = [4.60, 1.60, 0.69, 0.35, 0.11, 0.0, -0.69, -1.1, -1.4, -1.61, -1.79, -2.3, -2.4,
#     -2.49, -2.57, -2.64, -2.71, -2.77, -2.83, -2.89, -2.94, -2.996]
# print(y)
#
# showXandYplot(x,y, 'x', 'x and y')
#
# # Show raw x and y relationship
# dfX = pd.DataFrame({"x": x})
# dfY = pd.DataFrame({"y": y})
# dfX = sm.add_constant(dfX)
#
# # Show residuals from y(x)
# model       = sm.OLS(y, dfX).fit()
# predictions = model.predict(dfX)
# print(model.summary())
# showResidualPlotAndRMSE(x,y,predictions)
#
# # Section B: Transform and plot graph with transformed x.
# dfX['xt'] = -np.log(dfX['x'])
# showXandYplot(dfX['xt'] ,y, '-log(x)', '-log(x) vs. y')
#
# model_t       = sm.OLS(y, dfX[['const', 'xt']]).fit()
# predictions_t = model_t.predict(dfX[['const', 'xt']])
# print(model_t.summary())
# showResidualPlotAndRMSE(dfX['xt'], y, predictions_t)



# Exercise 2

# import matplotlib.pyplot    as plt
# from sklearn.metrics import mean_squared_error
# import pandas               as pd
# import statsmodels.api      as sm
# import numpy                as np
#
# #------------------------------------------------
# # Shows plot of x vs. y.
# #------------------------------------------------
# def showXandYplot(x,y, xtitle, title):
#     plt.figure(figsize=(8, 4))
#     plt.plot(x,y,color='blue')
#     plt.title(title)
#     plt.xlabel(xtitle)
#     plt.ylabel('y')
#     plt.show()
#
# #------------------------------------------------
# # Shows plot of actual vs. predicted and RMSE.
# #------------------------------------------------
# def showResidualPlotAndRMSE(x, y, predictions):
#     xmax      = max(x)
#     xmin      = min(x)
#     residuals = y - predictions
#
#     plt.figure(figsize=(8, 3))
#     plt.title('x and y')
#     plt.plot([xmin,xmax],[0,0],'--',color='black')
#     plt.title("Residuals")
#     plt.scatter(x,residuals,color='red')
#     plt.show()
#
#     # Calculate RMSE
#     mse = mean_squared_error(y,predictions)
#     rmse = np.sqrt(mse)
#     print("RMSE: " + str(rmse))
#
# # Section A: Define the raw data.
# x = [0.01, 0.2, 0.5, 0.7, 0.9,1,2,3,4,5,6,10,11,12,13,14,15,16,17,18,19,20]
# y = [0.99, 0.819, 0.6065, 0.4966, 0.40657, 0.368, 0.1353, 0.0498,
#      0.01831, 0.00674, 0.0025, 4.5399e-05, 1.670e-05, 6.1e-06,
#      2.260e-06, 8.3153e-07, 3.0590e-07, 1.12e-07, 4.14e-08,
#      1.52e-08, 5.60e-09, 2.061e-09]
# print(y)
#
# showXandYplot(x,y, 'x', 'x and y')
#
# # Show raw x and y relationship
# dfX = pd.DataFrame({"x": x})
# dfY = pd.DataFrame({"y": y})
# dfX = sm.add_constant(dfX)
#
# # Show residuals from y(x)
# model       = sm.OLS(y, dfX).fit()
# predictions = model.predict(dfX)
# print(model.summary())
# showResidualPlotAndRMSE(x,y,predictions)
#
# # Section B: Transform and plot graph with transformed x.
# dfX['xt'] = np.exp(-dfX['x'])
# showXandYplot(dfX['xt'] ,y, 'exp(-x)', 'exp(-x) vs. y')
#
# model_t       = sm.OLS(y, dfX[['const', 'xt']]).fit()
# predictions_t = model_t.predict(dfX[['const', 'xt']])
# print(model_t.summary())
# showResidualPlotAndRMSE(dfX['xt'], y, predictions_t)



# Exercise 3

import pandas               as pd
import statsmodels.api      as sm
import numpy                as np
import matplotlib.pyplot    as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#------------------------------------------------
# Shows plot of x vs. y.
#------------------------------------------------
def showXandYplot(x,y, xtitle, title):
    plt.figure(figsize=(8, 4))
    plt.plot(x,y,color='blue')
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel('y')
    plt.show()

PATH   = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
FILE   = 'abs.csv'
df     = pd.read_csv(PATH + FILE)

x = df[['abs(450nm)']]  # absorbance
y = df[['ug/L']]        # protein concentration
showXandYplot(x,y, 'absorbance x', 'Protein Concentration(y) and Absorbance(x)')

# Show raw x and y relationship
x = sm.add_constant(x)

# Show model.
model       = sm.OLS(y, x).fit()
predictions = model.predict(x)
print(model.summary())

# Show RMSE.
preddf      = pd.DataFrame({"predictions":predictions})
residuals   = y['ug/L']-preddf['predictions']
resSq       = [i**2 for i in residuals]
rmse        = np.sqrt(np.sum(resSq)/len(resSq))
print("RMSE: " + str(rmse))

# Show the residual plot
plt.scatter(x['abs(450nm)'],residuals)
plt.show()

x = df['abs(450nm)']
dfX = pd.DataFrame({"x": x})
dfX = sm.add_constant(dfX)

def grid_search(dfX, y, trans):
    trans_func = {
        'sqrt': lambda x: np.sqrt(x),
        'inv': lambda x: 1 / x,
        'neg_inv': lambda x: -1 / x,
        'sqr': lambda x: x * x,
        'log': lambda x: np.log(x),
        'neg_log': lambda x: -np.log(x),
        'exp': lambda x: np.exp(x),
        'neg_exp': lambda x: np.exp(-x),
    }
    dfTransformations = pd.DataFrame()
    for tran in trans:
        # Transform x
        dfX['xt'] = trans_func[tran](dfX['x'])
        model_t = sm.OLS(y, dfX[['const', 'xt']]).fit()
        predictions_t = model_t.predict(dfX[['const', 'xt']])
        # Calculate RMSE
        mse = mean_squared_error(y, predictions_t)
        rmse = np.sqrt(mse)
        dfTransformations = dfTransformations._append({
            "tran":tran, "rmse":rmse}, ignore_index=True)
    dfTransformations = dfTransformations.sort_values(by=['rmse'])
    return dfTransformations
rmsedf = grid_search(dfX, y, ('sqrt', 'neg_inv', 'log', 'exp', 'neg_exp'))
print(rmsedf)

def showResidualPlotAndRMSE(x, y, predictions):
    xmax      = max(x)
    xmin      = min(x)
    residuals = y - predictions

    plt.figure(figsize=(8, 3))
    plt.title('x and y')
    plt.plot([xmin,xmax],[0,0],'--',color='black')
    plt.title("Residuals")
    plt.scatter(x,residuals,color='red')
    plt.show()

    # Calculate RMSE
    mse = mean_squared_error(y,predictions)
    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))

dfX['xt'] = np.exp(dfX['x'])
x = dfX
x = sm.add_constant(x)

# Show model.
model       = sm.OLS(y, x).fit()
predictions = model.predict(x)
print(model.summary())

# Show RMSE.
preddf      = pd.DataFrame({"predictions":predictions})
residuals   = y['ug/L']-preddf['predictions']
resSq       = [i**2 for i in residuals]
rmse        = np.sqrt(np.sum(resSq)/len(resSq))
print("RMSE: " + str(rmse))
showResidualPlotAndRMSE(x['xt'], y['ug/L'], predictions)