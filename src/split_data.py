
import pandas as pd, os

data_interim = os.path.join("data", "interim", "clean_data.parquet")
train_path = os.path.join("data", 'processed', 'train.parquet')
test_path = os.path.join("data", 'processed', 'test.parquet')

data = pd.read_parquet(data_interim)

split = 0.30
n_train = int(len(data) - len(data) * split)

data[:n_train].reset_index(drop=True).to_parquet(train_path, compression="snappy")
data[n_train:].reset_index(drop=True).to_parquet(test_path, compression="snappy")

print("File Partitioned Successfully!")
