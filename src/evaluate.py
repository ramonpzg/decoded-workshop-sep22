
import pandas as pd, os, numpy as np, pickle, json, pprint
import sklearn.metrics as metrics

model_path = os.path.join('models', 'rf_model.pkl')
metrics_path = os.path.join("reports", 'metrics.json')
test_path = os.path.join("data", 'processed', 'test.parquet')

with open(model_path, "rb") as fd:
    model = pickle.load(fd)

X_test = pd.read_parquet(test_path)
y_test = X_test.pop('rented_bike_count')

predictions = model.predict(X_test.values)

mae = metrics.mean_absolute_error(y_test.values, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test.values, predictions))
r2_score = model.score(X_test.values, y_test.values)

with open(metrics_path, "w") as fd:
    json.dump({"MAE": mae, "RMSE": rmse, "R^2": r2_score}, fd, indent=4)

print("File Evaluated Successfully!")
pprint.pprint({"MAE": mae, "RMSE": rmse, "R^2": r2_score})
