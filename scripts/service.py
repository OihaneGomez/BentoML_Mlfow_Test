import numpy as np
import bentoml
import pickle
import os
from bentoml.io import NumpyNdarray

# 🔹 Path to the latest model downloaded from GitHub
MODEL_FILE = "/home/oihane/00_ToNoWaste/BentoML/Test_Automatic/models/latest/model.pkl"

# 🔹 Load model from `models/latest/`
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded from models/latest/")
else:
    raise FileNotFoundError(f"🚨 Model file not found at {MODEL_FILE}")

# 🔹 Save model to BentoML
bento_model = bentoml.sklearn.save_model("flower_model", model)

# 🔹 Get runner
runner = bentoml.sklearn.get(bento_model.tag).to_runner()

svc = bentoml.Service("clasificador_iris", runners=[runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    prediction = await runner.predict.async_run(input_series)
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    return classes[prediction]
