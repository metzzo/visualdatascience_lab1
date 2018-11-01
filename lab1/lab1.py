import pandas as pd


def load_dataset(path="../../nutritions.csv"):
    raw_dataset = pd.read_csv(
        filepath_or_buffer=path,
        sep=";",  # ; used to separate
        header=1,  # 1 row at the beginning is header
        doublequote=True
    )
    print(raw_dataset.columns)
    return raw_dataset

ds = load_dataset()
print(ds)