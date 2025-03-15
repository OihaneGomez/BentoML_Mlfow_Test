import os
import time
import subprocess
import pickle
import bentoml
import json
import requests
from git import Repo
from datetime import datetime

# 🔹 GitHub Repository URL
GITHUB_REPO = "https://github.com/OihaneGomez/BentoML_Mlfow_Test.git"
GITHUB_RAW_MODEL_URL = "https://raw.githubusercontent.com/OihaneGomez/BentoML_Mlfow_Test/main/models/latest/model.pkl"
GITHUB_RAW_METADATA_URL = "https://raw.githubusercontent.com/OihaneGomez/BentoML_Mlfow_Test/main/models/latest/metadata.json"

# 🔹 Local paths
LOCAL_REPO_PATH = "/home/oihane/00_ToNoWaste/BentoML/Test_Automatic"
MODEL_DIR = os.path.join(LOCAL_REPO_PATH, "models/latest")
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

def pull_latest_repo():
    """Fetch the latest changes from GitHub without overwriting local files."""
    print("📥 Checking GitHub for updates...")
    
    if os.path.exists(LOCAL_REPO_PATH):
        repo = Repo(LOCAL_REPO_PATH)
        repo.remotes.origin.fetch()  # Fetch latest updates without auto-merging
    else:
        Repo.clone_from(GITHUB_REPO, LOCAL_REPO_PATH)

    print("✅ GitHub repository checked for updates.")

def fetch_github_metadata():
    """Download metadata.json from GitHub to compare versions."""
    try:
        response = requests.get(GITHUB_RAW_METADATA_URL)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"⚠️ Could not fetch metadata.json from GitHub. HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"🚨 Error fetching metadata.json: {e}")
        return None

def has_model_changed():
    """Compare the local and GitHub versions to determine if an update is needed."""
    
    # Get local metadata
    if not os.path.exists(METADATA_FILE):
        print("🆕 No local metadata found. Assuming a new model is needed.")
        return True

    with open(METADATA_FILE, "r") as f:
        local_metadata = json.load(f)
    local_version = local_metadata.get("version", "0")

    # Get GitHub metadata
    github_metadata = fetch_github_metadata()
    if not github_metadata:
        print("⚠️ Skipping update check due to missing GitHub metadata.")
        return False

    github_version = github_metadata.get("version", "0")

    print(f"🔍 Local version: {local_version}, GitHub version: {github_version}")

    if int(github_version) > int(local_version):
        print("🔄 Newer model version detected! Updating...")
        return True

    print("✅ Local version is up-to-date or newer. No update needed.")
    return False

def download_model():
    """Download the latest model.pkl from GitHub if needed."""
    
    print("📥 Downloading model from GitHub...")
    
    try:
        response = requests.get(GITHUB_RAW_MODEL_URL, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
            with open(MODEL_FILE, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)

            print(f"✅ Model successfully downloaded to {MODEL_FILE}")
            return True
        else:
            print(f"🚨 Failed to download model. HTTP Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"🚨 Error downloading model: {e}")
        return False

def download_metadata():
    """Download the latest metadata.json from GitHub if needed."""
    
    print("📥 Downloading metadata.json from GitHub...")
    
    try:
        response = requests.get(GITHUB_RAW_METADATA_URL)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
            with open(METADATA_FILE, "w") as file:
                file.write(response.text)

            print(f"✅ Metadata successfully downloaded to {METADATA_FILE}")
            return True
        else:
            print(f"🚨 Failed to download metadata.json. HTTP Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"🚨 Error downloading metadata.json: {e}")
        return False

def update_bento_model():
    """Ensure the latest model and metadata exist and update BentoML dynamically."""
    pull_latest_repo()

    # 🔹 Step 1: Check if a new model version is available
    if not has_model_changed():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} 🔹 No new model found. Skipping update.")
        return

    # 🔹 Step 2: Download both the model and metadata from GitHub
    if not download_model():
        print("🚨 Model download failed. Skipping update.")
        return

    if not download_metadata():
        print("🚨 Metadata download failed. Skipping update.")
        return

    # 🔹 Step 3: Load and update BentoML with the new model
    print(f"📥 Loading model from {MODEL_FILE}")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    # 🔹 Step 4: Save model to BentoML
    bento_model = bentoml.sklearn.save_model("flower_model", model)
    print(f"✅ Updated BentoML Model: {bento_model.tag}")

    # 🔹 Step 5: Restart BentoML API to reload the new model
    print("🔄 Restarting BentoML API...")
    subprocess.run(["pkill", "-f", "bentoml serve"], stderr=subprocess.DEVNULL)
    subprocess.Popen(["bentoml", "serve", "/home/oihane/00_ToNoWaste/BentoML/Test_Automatic/scripts/service.py:svc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def continuous_check():
    """Continuously check for model updates every 5 minutes."""
    print("🔄 Starting GitHub model update service...")
    while True:
        update_bento_model()
        print("⏳ Waiting 5 minutes before checking again...")
        time.sleep(300)

if __name__ == "__main__":
    continuous_check()
