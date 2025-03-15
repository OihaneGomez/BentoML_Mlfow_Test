import mlflow
import bentoml
import os
import json
import time
import subprocess

mlflow.set_tracking_uri("http://localhost:5000")

def get_latest_production_model():
    client = mlflow.tracking.MlflowClient()

    try:
        latest_version = client.get_latest_versions("iris_classifier", stages=["Production"])[0]
        print(f"📥 Latest Production Model: Version {latest_version.version}")
        return latest_version.version
    except Exception as e:
        print("❌ No production model found in MLflow.")
        return None

def update_bento_model():
    latest_version = get_latest_production_model()
    if not latest_version:
        return

    model_uri = f"models:/iris_classifier/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Save model to BentoML
    bento_model = bentoml.sklearn.save_model("flower_model", model)
    print(f"✅ Updated BentoML Model: {bento_model.tag}")

    # Restart BentoML API
    print("🔄 Restarting BentoML API...")
    subprocess.run(["pkill", "-f", "bentoml serve"], stderr=subprocess.DEVNULL)
    subprocess.run(["bentoml", "serve", "service.py:svc", "&"])

def continuous_check():
    print("🔄 Starting MLflow model update service...")
    while True:
        update_bento_model()
        print("⏳ Waiting 5 minutes before checking again...")
        time.sleep(300)

if __name__ == "__main__":
    continuous_check()
