
import pandas as pd, os, numpy as np, pickle
from sklearn.ensemble import RandomForestRegressor

train_path = os.path.join("data", 'processed', 'train.parquet')

X_train = pd.read_parquet(train_path)
y_train = X_train.pop('rented_bike_count')

seed = 42
n_est = 100

rf = RandomForestRegressor(n_estimators=n_est, random_state=seed)
rf.fit(X_train.values, y_train.values)

with open(os.path.join('models', 'rf_model.pkl'), "wb") as fd:
    pickle.dump(rf, fd)
    
print("File Trained Successfully!")
