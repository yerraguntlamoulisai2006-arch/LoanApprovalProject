import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = pd.read_csv("dataset/loan.csv")

# 2. Fill missing values
data["Gender"] = data["Gender"].fillna(data["Gender"].mode()[0])
data["Married"] = data["Married"].fillna(data["Married"].mode()[0])
data["Dependents"] = data["Dependents"].fillna(data["Dependents"].mode()[0])
data["Self_Employed"] = data["Self_Employed"].fillna(data["Self_Employed"].mode()[0])

data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())
data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median())
data["Credit_History"] = data["Credit_History"].fillna(data["Credit_History"].mode()[0])

# 3. Convert Dependents
data["Dependents"] = data["Dependents"].replace("3+", 3)
data["Dependents"] = data["Dependents"].astype(int)

# 4. Convert text to numbers
encoder = LabelEncoder()

data["Gender"] = encoder.fit_transform(data["Gender"])
data["Married"] = encoder.fit_transform(data["Married"])
data["Education"] = encoder.fit_transform(data["Education"])
data["Self_Employed"] = encoder.fit_transform(data["Self_Employed"])
data["Property_Area"] = encoder.fit_transform(data["Property_Area"])
data["Loan_Status"] = encoder.fit_transform(data["Loan_Status"])

# 5. Separate input and output
X = data.drop(["Loan_ID", "Loan_Status"], axis=1)
y = data["Loan_Status"]

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

# 9. Take applicant details
print("\n--- Loan Application ---")

gender = input("Gender (Male/Female): ")
married = input("Married (Yes/No): ")
dependents = input("Dependents (0/1/2/3): ")
education = input("Education (Graduate/Not Graduate): ")
self_employed = input("Self Employed (Yes/No): ")

applicant_income = float(input("Applicant Income: "))
coapplicant_income = float(input("Coapplicant Income: "))
loan_amount = float(input("Loan Amount: "))
loan_term = float(input("Loan Amount Term: "))

credit_history = float(input("Credit History (1=Good, 0=Bad): "))

property_area = input("Property Area (Urban/Rural/Semiurban): ")

# 10. Convert applicant details
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

if property_area == "Rural":
    property_area = 0
elif property_area == "Semiurban":
    property_area = 1
else:
    property_area = 2

# 11. Create applicant data
applicant = [[
    gender,
    married,
    int(dependents),
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
]]

# 12. Predict
prediction = model.predict(applicant)

# 13. Display result
if prediction[0] == 1:
    print("\n🎉 LOAN APPROVED!")
else:
    print("\n❌ LOAN REJECTED!")