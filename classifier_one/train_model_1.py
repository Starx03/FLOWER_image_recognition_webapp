import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. LOAD THE DATASET
# (Make sure loan_data.csv is in the exact same folder!)
df = pd.read_csv('loan_approval_dataset.csv')

# Strip leading/trailing spaces from column names (common in CSVs)
df.columns = df.columns.str.strip()

print("--- DATASET PREVIEW ---")
print(df.head())
print("\n--- COLUMNS IN DATASET ---")
print(df.columns.tolist())

# 2. DROP USELESS COLUMNS
# Loan ID doesn't help predict loan eligibility
if 'loan_id' in df.columns:
    df = df.drop(columns=['loan_id'])

# 3. CLEAN UP TEXT VALUES AND CONVERT CATEGORIES TO NUMBERS
# Strip extra spaces in string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

# Map binary categorical text columns to 1s and 0s
if 'education' in df.columns:
    df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})

if 'self_employed' in df.columns:
    df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})

if 'loan_status' in df.columns:
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

# Drop any remaining unhandled missing values
df = df.dropna()

# 4. SEPARATE FEATURES (X) AND TARGET (y)
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# 5. SPLIT DATA INTO TRAIN & TEST SETS (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. TRAIN THE CLASSIFIER MODEL
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. EVALUATE MODEL PERFORMANCE
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n✅ Model Training Complete!")
print(f"Model Accuracy on Test Data: {accuracy:.2f}%")

# 8. SAVE THE TRAINED MODEL TO A FILE
joblib.dump(model, 'loan_model.pkl')
print("Model saved as 'loan_model.pkl' ")