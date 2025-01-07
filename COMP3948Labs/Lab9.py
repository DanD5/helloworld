# exercise 2

# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
#
# # Import data into a DataFrame.
# path = "/Users/dandidac/Documents/COMP 3948/Data Sets/bodyfat.txt"
# df = pd.read_csv(path, sep='\t')
# plt.rcParams.update({'font.size': 22})
#
# print(df.head())
# plt.subplots(nrows=1, ncols=4,  figsize=(14,7))
#
# plt.subplot(1 ,4, 1)
# boxplot = df.boxplot(column=['Pct.BF'])
#
# plt.subplot(1 ,4, 2)
# boxplot = df.boxplot(column=['Age'])
#
# plt.subplot(1 ,4, 3)
# boxplot = df.boxplot(column=['Weight'])
#
# plt.subplot(1 ,4, 4)
# boxplot = df.boxplot(column=['Height'])
#
# plt.show()




# Exercise 3

# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Import data into a DataFrame.
# path = "/Users/dandidac/Documents/COMP 3948/Data Sets/babysamp-98.txt"
# df = pd.read_csv(path, sep='\t')
#
# # Rename the columns so they are more reader-friendly.
# df = df.rename({'MomAge': 'Mom Age', 'DadAge':'Dad Age',
#                 'MomEduc':'Mom Edu', 'weight':'Weight'}, axis=1)  # new method
#
# dfSub = df[['Mom Age', 'Dad Age', 'Mom Edu', 'MomMarital', 'numlive', "dobmm", 'gestation']]
#
# from scipy import stats
# import numpy as np
# z = np.abs(stats.zscore(dfSub))
# print(z)
#
# THRESHOLD = 2.33
# print(np.where(z > THRESHOLD))
# print(dfSub.loc[39][[0]])
# print("Mom Age:", df["Mom Age"].loc[0])
# print("Dad Age:", df["Dad Age"].loc[0])
# print("Mom Edu:", df["Mom Edu"].loc[0])
# print("MomMarital:", df["MomMarital"].loc[0])
# print("numlive:", df["numlive"].loc[0])
# print("dobmm:", df["dobmm"].loc[0])
# print("gestation:", df["gestation"].loc[0])



# Exercise 4

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import statsmodels.api       as sm
# import matplotlib.pyplot     as plt
# from   scipy                 import stats
# import numpy                 as np
#
# PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "USA_Housing.csv"
# dataset = pd.read_csv(PATH + CSV_DATA)
#
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# print(dataset.head())
#
# # ------------------------------------------------------------------
# # Show statistics, boxplot, extreme values and returns DataFrame
# # row indexes where outliers exist.
# # ------------------------------------------------------------------
# def viewAndGetOutliers(df, colName, threshold, plt):
#     # Show basic statistics.
#     dfSub = df[[colName]]
#     print("*** Statistics for " + colName)
#     print(dfSub.describe())
#
#     # Show boxplot.
#     dfSub.boxplot(column=[colName])
#     plt.title(colName)
#     plt.show()
#
#     # Note this is absolute 'abs' so it gets both high and low values.
#     z = np.abs(stats.zscore(dfSub))
#     rowColumnArray = np.where(z > threshold)
#     rowIndices     = rowColumnArray[0]
#
#     # Show outlier rows.
#     print("\nOutlier row indexes for " + colName + ":")
#     print(rowIndices)
#     print("")
#
#     # Show filtered and sorted DataFrame with outliers.
#     dfSub = df.iloc[rowIndices]
#     dfSorted = dfSub.sort_values([colName], ascending=[True])
#     print("\nDataFrame rows containing outliers for " + colName + ":")
#     print(dfSorted)
#     return rowIndices
#
# THRESHOLD_Z      = 3
# priceOutlierRows = viewAndGetOutliers(dataset, 'Avg. Area Income', THRESHOLD_Z, plt)



# Exercise 5

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import statsmodels.api       as sm
# import matplotlib.pyplot     as plt
# from   scipy                 import stats
# import numpy                 as np
#
# path = "/Users/dandidac/Documents/COMP 3948/Data Sets/babysamp-98.txt"
# dataset = pd.read_csv(path, sep='\t')
#
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# print(dataset.head())
#
# # ------------------------------------------------------------------
# # Show statistics, boxplot, extreme values and returns DataFrame
# # row indexes where outliers exist.
# # ------------------------------------------------------------------
# def viewAndGetOutliers(df, colName, threshold, plt):
#     # Show basic statistics.
#     dfSub = df[[colName]]
#     print("*** Statistics for " + colName)
#     print(dfSub.describe())
#
#     # Show boxplot.
#     dfSub.boxplot(column=[colName])
#     plt.title(colName)
#     plt.show()
#
#     # Note this is absolute 'abs' so it gets both high and low values.
#     z = np.abs(stats.zscore(dfSub))
#     rowColumnArray = np.where(z > threshold)
#     rowIndices     = rowColumnArray[0]
#
#     # Show outlier rows.
#     print("\nOutlier row indexes for " + colName + ":")
#     print(rowIndices)
#     print("")
#
#     # Show filtered and sorted DataFrame with outliers.
#     dfSub = df.iloc[rowIndices]
#     dfSorted = dfSub.sort_values([colName], ascending=[True])
#     print("\nDataFrame rows containing outliers for " + colName + ":")
#     print(dfSorted)
#     return rowIndices
#
# THRESHOLD_Z      = 2.33
# priceOutlierRows = viewAndGetOutliers(dataset, 'weight', THRESHOLD_Z, plt)



# Exercise 6

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import statsmodels.api       as sm
# import matplotlib.pyplot     as plt
# from   scipy                 import stats
# import numpy                 as np
#
# path = "/Users/dandidac/Documents/COMP 3948/Data Sets/babysamp-98.txt"
# dataset = pd.read_csv(path, sep='\t')
#
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# print(dataset.head())
#
# # ------------------------------------------------------------------
# # Show statistics, boxplot, extreme values and returns DataFrame
# # row indexes where outliers exist outside an upper and lower percentile.
# # ------------------------------------------------------------------
# def viewAndGetOutliersByPercentile(df, colName, lowerP, upperP, plt):
#     # Show basic statistics.
#     dfSub = df[[colName]]
#     print("*** Statistics for " + colName)
#     print(dfSub.describe())
#
#     # Show boxplot.
#     dfSub.boxplot(column=[colName])
#     plt.title(colName)
#     plt.show()
#
#     # Get upper and lower perctiles and filter with them.
#     up = df[colName].quantile(upperP)
#     lp = df[colName].quantile(lowerP)
#     outlierDf = df[(df[colName] < lp) | (df[colName] > up)]
#
#     # Show filtered and sorted DataFrame with outliers.
#     dfSorted = outlierDf.sort_values([colName], ascending=[True])
#     print("\nDataFrame rows containing outliers for " + colName + ":")
#     print(dfSorted)
#
#     return lp, up  # return lower and upper percentiles
#
# LOWER_PERCENTILE = 0.02
# UPPER_PERCENTILE = 0.98
# lp, up = viewAndGetOutliersByPercentile(dataset, 'gestation',
#                                         LOWER_PERCENTILE, UPPER_PERCENTILE, plt)



# # Exercise 7
#
# import pandas as pd
#
# PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
# CSV_DATA = "wnba.csv"
# df = pd.read_csv(PATH + CSV_DATA,
#                       skiprows=1,  # Don't include header row as part of data.
#                       encoding="ISO-8859-1", sep=',', names=('PLAYER', 'GP', 'PTS'))
#
# df['GP'] = df['GP'].clip(0, 36)
# df['PTS'] = df['PTS'].clip(0, 860)
#
# print(df.head(30))



# Exercise 8

import pandas as pd

PATH = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
CSV_DATA = "wnba.csv"
df = pd.read_csv(PATH + CSV_DATA,
                 skiprows=1,  # Don't include header row as part of data.
                 encoding="ISO-8859-1", sep=',', names=('PLAYER', 'GP', 'PTS'))

nonOutlier_df = df[(df['GP'] <= 36) & (df['PTS'] < 860)]

print(nonOutlier_df.head(30))

