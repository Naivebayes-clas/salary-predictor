"""Load and clean the salary dataset."""
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "salary_data.csv"


def load_data(path: str = None) -> pd.DataFrame:
    """Load the salary dataset and perform basic cleaning."""
    path = Path(path) if path else DATA_PATH
    df = pd.read_csv(path)

    # Drop rows where salary is missing or zero
    df = df.dropna(subset=["salary"])
    df = df[df["salary"] > 0]

    # Remove duplicates
    df = df.drop_duplicates()

    return df  
