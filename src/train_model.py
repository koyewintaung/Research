# src/train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_and_save_model(data_path="data/sample_data.csv", model_path="models/model.joblib"):
    """
    Train a logistic regression model on the CVD dataset and save it as model.joblib
    """
    # 1. Load dataset
    df = pd.read_csv(data_path)

    # 2. Define features (X) and target (y)
    # ⚠️ Make sure the 'target' column exists in your dataset
    X = df.drop(columns=["patientid", "target"], errors="ignore")  
    y = df["target"]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Build a pipeline (scaling + logistic regression)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # 5. Train model
    pipeline.fit(X_train, y_train)

    # 6. Save model
    joblib.dump(pipeline, model_path)

    print(f"✅ Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
