import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Import data into a DataFrame.
path = "/Users/dandidac/Documents/COMP 3948/Data Sets/titanic_training_data.csv"
df   = pd.read_table(path, sep=',')
numericColumns = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch',
                  'Fare']

dfNumeric = df[numericColumns]
print(df.head())

# # Show data types for each columns.
print("\n*** Before imputing")
print(df.describe())
print(dfNumeric.head(11))

# Show summaries for objects like dates and strings.
from sklearn.impute import KNNImputer
imputer   = KNNImputer(n_neighbors=5)
dfNumeric = pd.DataFrame(imputer.fit_transform(dfNumeric),
                         columns = dfNumeric.columns)

# Show data types for each columns.
print("\n*** After imputing")
print(dfNumeric.describe())
print(dfNumeric.head(11))
