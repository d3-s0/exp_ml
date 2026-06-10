import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load data and encode target
df = pd.read_csv("data/certificates.csv").dropna(subset=["CURRENT_ENERGY_RATING"])

# Map letters (A-G) to ordered integers (0-6) so the ML model can understand the hierarchy
categories = [["A", "B", "C", "D", "E", "F", "G"]]
df["target"] = OrdinalEncoder(categories=categories).fit_transform(df[["CURRENT_ENERGY_RATING"]]).astype(int)

# Dynamic Feature Selection & Imputation
# Drop non-predictive metadata IDs to prevent the model from learning useless noise
exclude = ["UPRN", "BUILDING_REFERENCE_NUMBER", "LMK_KEY", "REPORT_TYPE"]
X_df = df.select_dtypes(include=["number"]).drop(columns=exclude, errors="ignore")

# Force everything to numbers and fill missing data with the median to avoid failing the model fit step
X_df = X_df.apply(lambda col: pd.to_numeric(col, errors="coerce").fillna(col.median())).dropna(axis=1, how="all")

# Train/Validation/Test Split (60/20/20)
# Split into Train and a temporary block (40%) to ensure unseen test data is completely isolated
X_train, X_temp, y_train, y_temp = train_test_split(X_df, df["target"], test_size=0.4, random_state=42)
# Split the temporary block in half to get exactly 20% validation and 20% test data
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Resample Train Data Only
# Duplicate minority class samples ONLY on training data to fix class imbalances 
# without leaking fake data into validation/test
X_train_res, y_train_res = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)

# Scale Features
# KNN relies on physical distance; scaling ensures large metrics 
# (e.g., square footage) don't overpower small ones (e.g., room count)
scaler = StandardScaler()
# Fit and transform training data to learn the mean/variance
X_train_scaled = scaler.fit_transform(X_train_res)
# Transform val/test datasets using the training scale parameters to prevent cheating (data leakage)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Model Training & Evaluation
# Fit the model using the balanced, scaled training inputs
model = KNeighborsClassifier(n_neighbors=1).fit(X_train_scaled, y_train_res)

# Print a breakdown of precision, recall, and f1-score to judge actual performance on unseen test data
print(classification_report(y_test, model.predict(X_test_scaled)))

