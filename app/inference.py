# app/inference.py
# Loads the trained artifacts ONCE at startup and scores single flows.
import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.features import FEATURES, build_matrix
from src.model import Autoencoder

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


class Detector:
    def __init__(self):
        self.meta = json.loads((ARTIFACTS / "metadata.json").read_text())
        self.columns = self.meta["feature_columns"]
        self.threshold = float(self.meta["threshold"])
        self.scaler = joblib.load(ARTIFACTS / "scaler.joblib")
        self.model = Autoencoder(self.meta["input_dim"], self.meta["bottleneck"])
        self.model.load_state_dict(torch.load(ARTIFACTS / "autoencoder.pth", map_location="cpu"))
        self.model.eval()

    def missing_features(self, flow: dict):
        return [c for c in FEATURES if c not in flow]

    def score(self, flow: dict):
        """flow = dict with the 41 raw NSL-KDD features -> (reconstruction_error, is_anomaly)."""
        row = pd.DataFrame([{c: flow[c] for c in FEATURES}])
        x = self.scaler.transform(build_matrix(row, self.columns)).astype("float32")
        with torch.no_grad():
            t = torch.from_numpy(x)
            err = float(((self.model(t) - t) ** 2).mean())
        return err, err > self.threshold
