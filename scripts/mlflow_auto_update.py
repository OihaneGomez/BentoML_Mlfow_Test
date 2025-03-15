import os
import time
import subprocess
import shutil
import pickle
import bentoml
from git import Repo

# 🔹 GitHub Repository URL (Replace with your correct username)
GITHUB_REPO = "https://github.com/OihaneGomez/BentoML_Mlfow_Test.git"
LOCAL_REPO_PATH = "/home/oihane/00_ToNoWaste/BentoML/Test_Automatic"

# 🔹 Paths for downloaded models
MODEL_DIR = os.path.join(LOCAL_REPO_PATH, "models/latest")
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
LAST_MODEL_HASH_FILE = "/tmp/last_model_hash.txt"

def pull_latest_repo():
    """Pull the latest model from GitHub."""
    print("📥 Fetching latest model from GitHub...")
    if os.path.exists(LOCAL_REPO_PATH):
        repo = Repo(LOCAL_REPO_PATH)
        repo.remotes.origin.pull()
    else:
        Repo.clone_from(GITHUB_REPO, LOCAL_REPO_PATH)
    print("✅ GitHub repository is up to date.")

def get_model_hash():
    """Generate a hash for the current model file to detect updates."""
    if not os.path.exists(MODEL_FILE):
        return None
    return str(os.path.getmtime(MODEL_FILE))  # Uses file modification time as a hash

def has_model_changed():
    """Check if a new model has been downloaded from GitHub."""
    current_hash = get_model_hash()
    if not current_hash:
        return False

    if os.path.exists(LAST_MODEL_HASH_FILE):
        with open(LAST_MODEL_HASH_FILE, "r") as f:
            last_hash = f.read().strip()
        if current_hash == last_hash:
            return False  # No new model

    with open(LAST_MODEL_HASH_FILE, "w") as f:
        f.write(current_hash)  # Save the new hash
    return True

def update_bento_model():
    """Load the latest model from GitHub and update BentoML dynamically."""
    if not os.path.exists(MODEL_FILE):
        print("❌ No model found in models/latest/. Skipping update.")
        return

    if not has_model_changed():
        print("🔹 No new model found. Skipping update.")
        return

    # 🔹 Load model from the downloaded GitHub repo
    print(f"📥 Loading model from {MODEL_FILE}")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    # 🔹 Save model to BentoML
    bento_model = bentoml.sklearn.save_model("flower_model", model)
    print(f"✅ Updated BentoML Model: {bento_model.tag}")

    # 🔄 Restart BentoML API to reload the new model
    print("🔄 Restarting BentoML API...")
    subprocess.run(["pkill", "-f", "bentoml serve"], stderr=subprocess.DEVNULL)
    subprocess.Popen(["bentoml", "serve", "/home/oihane/00_ToNoWaste/BentoML/Test_Automatic/scripts/service.py:svc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def continuous_check():
    """Continuously check for model updates every 5 minutes."""
    print("🔄 Starting GitHub model update service...")
    while True:
        pull_latest_repo()
        update_bento_model()
        print("⏳ Waiting 5 minutes before checking again...")
        time.sleep(300)

if __name__ == "__main__":
    continuous_check()
