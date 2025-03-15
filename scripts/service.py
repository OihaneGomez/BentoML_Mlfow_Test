import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

# Get the scikit-learn model from BentoML store
runner = bentoml.sklearn.get("flower_model:latest").to_runner()

svc = bentoml.Service("clasificador_iris", runners=[runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    prediction = await runner.predict.async_run(input_series)
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    return classes[prediction]

