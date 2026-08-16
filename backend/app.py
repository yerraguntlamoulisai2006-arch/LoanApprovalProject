from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sqlite3
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "loan.csv"
)

DB_FILE = os.path.join(
    BASE_DIR,
    "backend",
    "loan_applications.db"
)


# =========================================================
# DATABASE
# =========================================================

def create_database():

    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            bank_name TEXT,

            gender TEXT,
            married TEXT,
            dependents TEXT,
            education TEXT,
            self_employed TEXT,

            applicant_income REAL,
            coapplicant_income REAL,
            loan_amount REAL,
            loan_term REAL,

            credit_history TEXT,
            property_area TEXT,

            prediction TEXT,
            probability REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # -----------------------------------------------------
    # ADD bank_name TO OLD DATABASE IF IT DOES NOT EXIST
    # -----------------------------------------------------

    cursor.execute(
        "PRAGMA table_info(applications)"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "bank_name" not in columns:

        cursor.execute("""
            ALTER TABLE applications
            ADD COLUMN bank_name TEXT
        """)

        print(
            "Added bank_name column to existing database."
        )

    connection.commit()
    connection.close()

    print("Database ready!")


# =========================================================
# LOAD DATASET
# =========================================================

print("Loading dataset...")

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )


data = pd.read_csv(DATA_FILE)

print(
    "Dataset loaded successfully!"
)

print(
    "Dataset shape:",
    data.shape
)


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

data.columns = (
    data.columns
    .str.strip()
)


# =========================================================
# RENAME DATASET COLUMN
# =========================================================

if "Loan_Amount_Term" in data.columns:

    data = data.rename(
        columns={
            "Loan_Amount_Term":
                "LoanTerm"
        }
    )


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

features = [

    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",

    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "LoanTerm",

    "Credit_History",
    "Property_Area"
]


required_columns = features + [
    "Loan_Status"
]


missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing_columns:

    raise ValueError(
        "Missing dataset columns: "
        + str(missing_columns)
    )


# =========================================================
# CLEAN TARGET
# =========================================================

data["Loan_Status"] = data[
    "Loan_Status"
].map({
    "Y": 1,
    "N": 0
})


# =========================================================
# CLEAN DEPENDENTS
# =========================================================

data["Dependents"] = data[
    "Dependents"
].replace(
    "3+",
    "3"
)


# =========================================================
# X AND Y
# =========================================================

X = data[features]

y = data["Loan_Status"]


print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

categorical_features = [

    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]


# =========================================================
# NUMERICAL FEATURES
# =========================================================

numeric_features = [

    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "LoanTerm",
    "Credit_History"
]


# =========================================================
# NUMERIC PIPELINE
# =========================================================

numeric_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    )
])


# =========================================================
# CATEGORICAL PIPELINE
# =========================================================

categorical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


# =========================================================
# PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer([

    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# =========================================================
# MODEL
# =========================================================

model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=5000
        )
    )
])


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# =========================================================
# TRAIN
# =========================================================

print(
    "Training model..."
)

model.fit(
    X_train,
    y_train
)


# =========================================================
# ACCURACY
# =========================================================

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    "Model Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)

print(
    "ML model trained successfully!"
)


# =========================================================
# CREATE DATABASE
# =========================================================

create_database()


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return jsonify({

        "success": True,

        "message":
            "Loan Approval ML API is running!",

        "accuracy":
            round(
                accuracy * 100,
                2
            )

    })


# =========================================================
# PREDICT
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        request_data = request.get_json()

        print("")
        print(
            "Received application:"
        )

        print(
            request_data
        )


        # =================================================
        # BANK NAME
        # =================================================

        bank_name = request_data.get(
            "bank_name",
            "N/A"
        )


        if not bank_name:

            bank_name = "N/A"


        # =================================================
        # OTHER VALUES
        # =================================================

        gender = request_data.get(
            "gender",
            "Male"
        )

        married = request_data.get(
            "married",
            "No"
        )

        dependents = request_data.get(
            "dependents",
            "0"
        )

        education = request_data.get(
            "education",
            "Graduate"
        )

        self_employed = request_data.get(
            "self_employed",
            "No"
        )

        applicant_income = float(
            request_data.get(
                "applicant_income",
                0
            )
        )

        coapplicant_income = float(
            request_data.get(
                "coapplicant_income",
                0
            )
        )

        loan_amount = float(
            request_data.get(
                "loan_amount",
                0
            )
        )

        loan_term = float(
            request_data.get(
                "loan_term",
                360
            )
        )

        credit_history = request_data.get(
            "credit_history",
            "1"
        )

        property_area = request_data.get(
            "property_area",
            "Urban"
        )


        # =================================================
        # CREDIT HISTORY
        # =================================================

        if str(
            credit_history
        ).lower() in [

            "good",
            "1",
            "yes"
        ]:

            credit_value = 1.0

        else:

            credit_value = 0.0


        # =================================================
        # DEPENDENTS
        # =================================================

        if str(
            dependents
        ) == "3+":

            dependents = "3"


        # =================================================
        # APPLICANT DATA
        # =================================================

        applicant = pd.DataFrame([{

            "Gender":
                gender,

            "Married":
                married,

            "Dependents":
                str(dependents),

            "Education":
                education,

            "Self_Employed":
                self_employed,

            "ApplicantIncome":
                applicant_income,

            "CoapplicantIncome":
                coapplicant_income,

            "LoanAmount":
                loan_amount,

            "LoanTerm":
                loan_term,

            "Credit_History":
                credit_value,

            "Property_Area":
                property_area

        }])


        # =================================================
        # PREDICTION
        # =================================================

        result = model.predict(
            applicant
        )[0]


        probability = model.predict_proba(
            applicant
        )[0][1]


        if result == 1:

            status = "Approved"

        else:

            status = "Rejected"


        probability_percent = round(
            probability * 100,
            2
        )


        # =================================================
        # SAVE TO DATABASE
        # =================================================

        connection = sqlite3.connect(
            DB_FILE
        )

        cursor = connection.cursor()


        cursor.execute("""
            INSERT INTO applications (

                bank_name,

                gender,
                married,
                dependents,
                education,
                self_employed,

                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_term,

                credit_history,
                property_area,

                prediction,
                probability

            )

            VALUES (

                ?,

                ?, ?, ?, ?, ?,

                ?, ?, ?, ?,

                ?, ?,

                ?, ?

            )
        """, (

            bank_name,

            gender,
            married,
            dependents,
            education,
            self_employed,

            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,

            str(credit_history),
            property_area,

            status,
            probability

        ))


        # =================================================
        # APPLICATION ID
        # =================================================

        application_id = cursor.lastrowid


        connection.commit()
        connection.close()


        print(
            "Prediction:",
            status
        )

        print(
            "Approval Probability:",
            probability_percent,
            "%"
        )

        print(
            "Bank:",
            bank_name
        )

        print(
            "Application saved!"
        )

        print(
            "Application ID:",
            application_id
        )


        # =================================================
        # RESPONSE
        # =================================================

        return jsonify({

            "success": True,

            "prediction":
                status,

            "probability":
                probability_percent,

            "application_id":
                application_id,

            "bank_name":
                bank_name,

            "message":
                "Loan Approved!"
                if status == "Approved"
                else
                "Loan Rejected!"

        })


    except Exception as error:

        print(
            "ERROR:",
            str(error)
        )


        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# =========================================================
# ADMIN STATS
# =========================================================

@app.route(
    "/admin/stats",
    methods=["GET"]
)
def admin_stats():

    try:

        connection = sqlite3.connect(
            DB_FILE
        )

        cursor = connection.cursor()


        cursor.execute(
            "SELECT COUNT(*) FROM applications"
        )

        total = cursor.fetchone()[0]


        cursor.execute("""
            SELECT COUNT(*)
            FROM applications
            WHERE prediction = 'Approved'
        """)

        approved = cursor.fetchone()[0]


        cursor.execute("""
            SELECT COUNT(*)
            FROM applications
            WHERE prediction = 'Rejected'
        """)

        rejected = cursor.fetchone()[0]


        connection.close()


        if total > 0:

            approval_rate = (
                approved / total
            ) * 100

        else:

            approval_rate = 0


        return jsonify({

            "total":
                total,

            "approved":
                approved,

            "rejected":
                rejected,

            "approval_rate":
                round(
                    approval_rate,
                    2
                )

        })


    except Exception as error:

        return jsonify({

            "error":
                str(error)

        }), 500


# =========================================================
# ADMIN APPLICATIONS
# =========================================================

@app.route(
    "/admin/applications",
    methods=["GET"]
)
def admin_applications():

    try:

        connection = sqlite3.connect(
            DB_FILE
        )

        connection.row_factory = sqlite3.Row

        cursor = connection.cursor()


        cursor.execute("""
            SELECT
                id,
                bank_name,

                gender,
                married,
                dependents,
                education,
                self_employed,

                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_term,

                credit_history,
                property_area,

                prediction,
                probability,

                created_at

            FROM applications

            ORDER BY id DESC
        """)


        rows = cursor.fetchall()


        applications = [
            dict(row)
            for row in rows
        ]


        connection.close()


        return jsonify(
            applications
        )


    except Exception as error:

        print(
            "ADMIN APPLICATION ERROR:",
            str(error)
        )


        return jsonify({

            "error":
                str(error)

        }), 500


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    print("")
    print(
        "======================================"
    )

    print(
        "🏦 LOAN APPROVAL SYSTEM"
    )

    print(
        "======================================"
    )

    print(
        "ML Accuracy:",
        round(
            accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "API:",
        "http://127.0.0.1:5000"
    )

    print(
        "======================================"
    )

    print("")


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )