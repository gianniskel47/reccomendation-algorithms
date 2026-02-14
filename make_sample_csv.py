import numpy as np
import pandas as pd

data = np.load("Dataset.npy", allow_pickle=True)

sample = data[:5000]

rows = [x.split(",") for x in sample]

df = pd.DataFrame(rows, columns=["user", "item", "rating", "date"])

df.to_csv("Dataset_sample.csv", index=False)

print(" Sample CSV created: Dataset_sample.csv")
