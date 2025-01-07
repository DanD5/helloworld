import pickle
import pandas as pd

CSV_DATA = "VehicleInsuranceClaims_Mystery.csv"
df = pd.read_csv(CSV_DATA, skiprows=1, encoding="ISO-8859-1", sep=',',
                 names=("Maker", "Model", "Adv_year", "Adv_month", "Color", "Reg_year", "Bodytype", "Runned_Miles",
                        "Engin_size", "Gearbox", "Fuel_type", "Price", "Seat_num", "Door_num", "issue", "issue_id",
                        "Adv_day", "breakdown_date", "repair_complexity", "repair_cost", "repair_hours", "category_anomaly",
                        "repair_date"))

# Loading best model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

numerical_cols = df.select_dtypes(include=["number"]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Cleaning the 'Engin_size' column by removing 'L'
df["Engin_size"] = df["Engin_size"].str.replace("L", "", regex=False).astype(float)

# Removing slashes and convert to integers
df['breakdown_date'] = (df['breakdown_date'].fillna('0/0/0').astype(str).str.replace("/", "", regex=False).astype(int))
df['repair_date'] = (df['repair_date'].fillna('0/0/0').astype(str).str.replace("/", "", regex=False).astype(int))

# Converting categorical columns to dummy variables
df = pd.get_dummies(df, columns=["Maker", "Model", "Color", "Bodytype", "Gearbox", "Fuel_type", "issue"], dtype=int)

# Ensuring Model_Focus is there
required_dummies = ["Model_Focus"]
for col in required_dummies:
    if col not in df.columns:
        df[col] = 0

# Binning
bins_miles = [0, 50000, 100000, 200000, 500000, float('inf')]
labels_miles = ["0-50k", "50k-100k", "100k-200k", "200k-500k", "500k+"]
runned_miles_binned = pd.cut(df["Runned_Miles"], bins=bins_miles, labels=labels_miles)

new_columns = pd.DataFrame({
    "Runned_Miles_Binned": runned_miles_binned,
})

df = pd.concat([df, new_columns], axis=1)

# One-hot encoding
df = pd.get_dummies(df, columns=["Runned_Miles_Binned"], drop_first=True)

X = df[["category_anomaly", "repair_hours", "Seat_num", "Door_num", "Model_Focus"]]

# Predictions
predictions = model.predict(X)

predictions_df = pd.DataFrame(predictions, columns=["Claim"])

predictions_df["Claim"] = predictions_df["Claim"].round(4)

output_file = "VehicleInsuranceClaims_Predictions.csv"
predictions_df.to_csv(output_file, index=False)