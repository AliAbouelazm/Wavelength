from datasets import load_dataset
import pandas as pd


def load_spotify_data() -> pd.DataFrame:
    dataset = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
    return dataset.to_pandas()


if __name__ == "__main__":
    df = load_spotify_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(df.columns.tolist())
