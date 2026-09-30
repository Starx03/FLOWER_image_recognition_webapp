import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#1. CREATING A MINI DATA SET
#X= input features (hours studied, attendance)
#y= target variable (1= pass, 0= fail)
data={
    'hours_studied': [2,8,2,9,7,3,10,4,6,2],
    'attendance': [50, 90, 40, 95, 85, 60, 98, 70, 80, 55],
    'passed': [0,1,0,1,1,0,1,0,1,0]
}

df = pd.DataFrame(data)

#sepearate input =X and target variable =y
X = df[['hours_studied', 'attendance']]
y = df['passed']

# 2. SPLIT DATA INTO TRAIN & TEST SETS (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. CHOOSE AND TRAIN THE MODEL
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. EVALUATE THE MODEL
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.0f}%")

# 5. TEST IT WITH NEW, UNSEEN DATA
# Let's predict for a student who studied 7 hours and had 85% attendance
sample_student = [[7, 85]]
prediction = model.predict(sample_student)

if prediction[0] == 1:
    print("Prediction for new student: PASSED! 🎉")
else:
    print("Prediction for new student: FAILED. ❌")