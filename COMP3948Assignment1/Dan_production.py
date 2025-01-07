import pandas as pd
import statsmodels.api as sm
import pickle

# Read file
CSV_DATA = "grades_mystery.csv"
df = pd.read_csv(CSV_DATA, skiprows=1, encoding="ISO-8859-1", sep=',',
                 names=("school", "sex", "age", "address", "famsize", "Pstatus",
                        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime",
                        "studytime", "failures", "schoolsup", "famsup", "paid", "activities",
                        "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout",
                        "Dalc", "Walc", "health", "absences", "grade"))

# Load best model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Dealing with possible missing values
numerical_cols = df.select_dtypes(include=['number']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

X = df[["Medu", "failures", "goout", "age"]]
X = sm.add_constant(X)

# Make predictions
predictions = model.predict(X)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(predictions, columns=['grade'])

# Round the predictions to 4 decimal points
predictions_df['grade'] = predictions_df['grade'].round(4)

# Save the predictions to a CSV file
output_file = "grade_predict.csv"
predictions_df.to_csv(output_file, index=False)
