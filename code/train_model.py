"""
train_model.py
--------------
Trains a claim risk classification model (RandomForest) using a Malaysia-calibrated dataset.
Saves a scikit-learn Pipeline (preprocess + model) to risk_model.pkl

Expected CSV columns (example):
- risk_level (target)
- claim_id (optional id)
- claim_amount, state, incident_type, policy_type, etc.

Usage:
    python train_model.py
Output:
    risk_model.pkl
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = "claims_dataset_malaysia_calibrated_10000.csv"
TARGET_COL = "risk_level"
MODEL_OUT = "risk_model.pkl"


def main():
    # Load training data
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {df.columns.tolist()}")

    # Drop non-feature columns if present (IDs etc.)
    drop_cols = [c for c in ["claim_id"] if c in df.columns]

    X = df.drop(columns=drop_cols + [TARGET_COL])
    y = df[TARGET_COL]

    # Detect categorical vs numeric columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocessing: OneHotEncode categorical, pass-through numeric
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Model: RandomForest (balanced for class imbalance)
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # Full pipeline: preprocessing + model
    clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    # Train/test split (stratify keeps label distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, pred))

    # Save model pipeline
    joblib.dump(clf, MODEL_OUT)
    print(f"\nSaved model: {MODEL_OUT}")


if __name__ == "__main__":
    main()
