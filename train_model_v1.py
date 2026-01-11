import os
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# File paths
DATA_PATH = os.path.join("data", "heart_disease_uci.csv")
MODEL_PATH = os.path.join("models", "model_v1.pkl")

TARGET_COL = "num"   # heart disease label
DROP_COLS = ["id"]   # remove patient ID from features

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Drop ID column
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Convert target to binary: 0 = no disease, 1 = disease
    df["target_binary"] = (df[TARGET_COL] > 0).astype(int)
    df = df.drop(columns=[TARGET_COL])

    X = df.drop(columns=["target_binary"])
    y = df["target_binary"]

    # Separate numeric and categorical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Pipelines for preprocessing with imputation
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Model V1: Logistic Regression baseline
    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Model V1 - Logistic Regression (with Imputer)")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    dump(pipeline, MODEL_PATH)
    print(f"\nModel V1 saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()

