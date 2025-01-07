from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Import Data
PATH      = "/Users/dandidac/Documents/COMP 3948/Data Sets/"
FILE      = "carPrice.csv"
df        = pd.read_csv(PATH + FILE)

# Enable the display of all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#---------------------------------------------
# Generate quick views of data.
def viewQuickStats():
    print("\n*** Show contents of the file.")
    print(df.head())

    print("\n*** Show the description for all columns.")
    print(df.info())

    print("\n*** Describe numeric values.")
    print(df.describe())

    print("\n*** Showing frequencies.")

    # Show frequencies.
    print(df['model'].value_counts())
    print("")
    print(df['transmission'].value_counts())
    print("")
    print(df['fuel type'].value_counts())
    print("")
    print(df['engine size'].value_counts())
    print("")
    print(df['fuel type2'].value_counts())
    print("")
    print(df['year'].value_counts())
    print("")

#---------------------------------------------
# Fix the price column.
for i in range(0, len(df)):
    priceStr = str(df.iloc[i]['price'])
    priceStr = priceStr.replace("£", "")
    riceStr  = priceStr.replace("-", "")
    priceStr = priceStr.replace(",", "")
    df.at[i,'price'] = priceStr

# Convert column to number.
df['price'] = pd.to_numeric(df['price'])

#---------------------------------------------
# Fix the price column.
averageYear = df['year'].mean()
for i in range(0, len(df)):
    year = df.iloc[i]['year']

    if(np.isnan(year)):
        df.at[i, 'year'] = averageYear

#---------------------------------------------
# Fix the engine size2 column.
for i in range(0, len(df)):
    try:
        engineSize2 = df.loc[i]['engine size2']
        if(pd.isna(engineSize2)):
            df.at[i,'engine size2'] = "0"

    except Exception as e:
        error = str(e)
        print(error)

df['engine size2'] = pd.to_numeric(df['engine size2'])
df['mileage2'].value_counts()
viewQuickStats()

#---------------------------------------------
# Fix the mileage column.
for i in range(0, len(df)):
    mileageStr = str(df.iloc[i]['mileage'])
    mileageStr = mileageStr.replace(",", "")
    df.at[i, 'mileage'] = mileageStr
    try:
        if(not mileageStr.isnumeric()):
            df.at[i, 'mileage'] = "0"
    except Exception as e:
        error = str(e)
        print(error)

df['mileage'] = pd.to_numeric(df['mileage'])
viewQuickStats()

# Isolate columns
tempDf = df[['transmission', 'fuel type2']]
# Get dummies
dummyDf = pd.get_dummies(tempDf, columns=['transmission', 'fuel type2'], dtype=int)
# Join dummy df with original df
df = pd.concat(([df, dummyDf]), axis=1)
print(df.head())

# Create separate bin columns for specific years
years = [2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013]
for year in years:
    df[f'year_{year}'] = np.where(df['year'] == year, 1, 0)

# Create a column for anything less than 2013
df['year_less_than_2013'] = np.where(df['year'] < 2013, 1, 0)
print(df.head())

# Ensure all relevant columns are numeric
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['engine size2'] = pd.to_numeric(df['engine size2'], errors='coerce')
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

# Select only numeric columns for correlation matrix
df_numeric = df.select_dtypes(include=[np.number])

# Check for NaN values and handle them
df_numeric = df_numeric.fillna(0)  # Option to fill NaN with 0 or another strategy

# Compute the correlation matrix
corr = df_numeric.corr()

# Plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()

X = df[['engine size2', 'year'] + list(dummyDf.columns) + [f'year_{year}' for year in years] + ['year_less_than_2013']]

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn                 import metrics
import statsmodels.api       as sm
# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) # make the predictions by the model

print(model.summary())

print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))
