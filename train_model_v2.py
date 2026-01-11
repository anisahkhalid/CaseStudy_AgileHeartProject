import os
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

# File paths
DATA_PATH = os.path.join("data", "heart_disease_uci.csv")
MODEL_PATH = os.path.join("models", "model_v2.pkl")

TARGET_COL = "num"
DROP_COLS = ["id"]

def main():
    df = pd.read_csv(DATA_PATH)

    # Drop ID column
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Binary target: 0 = no disease, 1 = disease
    df["target_binary"] = (df[TARGET_COL] > 0).astype(int)
    df = df.drop(columns=[TARGET_COL])

    X = df.drop(columns=["target_binary"])
    y = df["target_binary"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
        # RandomForest does NOT require scaling
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

    # Model V2: Random Forest (improved)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("Model V2 - Random Forest (Improved)")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    dump(pipeline, MODEL_PATH)
    print(f"\nModel V2 saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
