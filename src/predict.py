"""Load the trained model and make predictions."""
import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"


def load_model():
    """Load the trained pipeline from disk."""
    return joblib.load(MODEL_PATH)


def predict(pipe, features: dict) -> float:
    """Make a salary prediction."""
    df = pd.DataFrame([features])
    prediction = pipe.predict(df)[0]
    return max(0, prediction)
  
