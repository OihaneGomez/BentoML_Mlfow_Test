import mlflow
import bentoml
import pickle
import json
import os
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define paths
MODEL_DIR = "models/latest"
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_register_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Ensure MLflow logs locally
    mlflow.set_experiment("iris_classification")

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"✅ Model trained with accuracy: {acc:.2f}")

    with mlflow.start_run() as run:
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            registered_model_name="iris_classifier"
        )
        mlflow.log_metric("accuracy", acc)

        # Move the model to "Production"
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="iris_classifier",
            version=model_info.version,
            stage="Production"
        )

    # Save model locally
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = {
        "version": str(model_info.version),
        "accuracy": acc,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    train_and_register_model()