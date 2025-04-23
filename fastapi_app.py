# from fastapi import FastAPI
# import pandas as pd
# import joblib
# import os

# app = FastAPI()

# # Preload all models from the "models/RandomF" directory
# model_dir = "models/RandomF"
# models = {
#     fname.replace("model_", "").replace(".joblib", ""): joblib.load(os.path.join(model_dir, fname))
#     for fname in os.listdir(model_dir) if fname.endswith(".joblib")
# }

# @app.get("/predict")
# def predict(input_data: dict):
#     features = ["RET_mean", "RET_std", "RET_skew", "RET_kurt", "RET_rf", "vwap", "RET_SPY"]
#     df = pd.DataFrame(input_data)
#     predictions = []

#     for symbol, model in models.items():
#         row = df[df["symbol"] == symbol]
#         if not row.empty:
#             pred = model.predict(row[features])[0]
#             predictions.append({"symbol": symbol, "prediction": pred})

#     # Sort and get top 25
#     top_25 = sorted(predictions, key=lambda x: x["prediction"], reverse=True)[:25]
#     return {"top_25": top_25}

from fastapi import FastAPI
from typing import List, Dict
import pandas as pd
import joblib
import os

app = FastAPI()

# Preload all models from the "models/RandomF" directory
model_dir = "models/RandomF"
models = {
    fname.replace("model_", "").replace(".joblib", ""): joblib.load(os.path.join(model_dir, fname))
    for fname in os.listdir(model_dir) if fname.endswith(".joblib")
}

@app.post("/predict")
def predict(input_data: List[Dict]):
    features = ["RET_mean", "RET_std", "RET_skew", "RET_kurt", "RET_rf", "vwap", "RET_SPY"]
    df = pd.DataFrame(input_data)

    predictions = []
    for symbol, model in models.items():
        row = df[df["symbol"] == symbol]
        if not row.empty:
            pred = model.predict(row[features])[0]
            predictions.append({"symbol": symbol, "prediction": float(pred)})

    top_25 = sorted(predictions, key=lambda x: x["prediction"], reverse=True)[:25]
    return {"top_25": top_25}
