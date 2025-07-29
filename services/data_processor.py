import pandas as pd

def process_data(file_path: str):
    df = pd.read_csv(file_path)
    stats = df.describe()
    return stats.to_dict()