
import pandas as pd, os, sys, re

raw_data = os.path.join("data", 'raw', "SeoulBikeData.csv")
data_interim = os.path.join("data", "interim", "clean_data.parquet")

data = pd.read_csv(raw_data, encoding='iso-8859-1')

def clean_col_names(list_of_cols):
    return [re.sub(r'[^a-zA-Z0-9\s]', '', col).lower().replace(r" ", "_") for col in list_of_cols]

def extract_dates(data):
    data['date'] = pd.to_datetime(data['date'], format="%d/%m/%Y")
    data.sort_values(['date', 'hour'], inplace=True)
    data["year"] = data['date'].dt.year
    data["month"] = data['date'].dt.month
    data["week"] = data['date'].dt.isocalendar().week
    data["day"] = data['date'].dt.day
    data["day_of_week"] = data['date'].dt.dayofweek
    data["day_of_year"] = data['date'].dt.dayofyear
    data["is_month_end"] = data['date'].dt.is_month_end
    data["is_month_start"] = data['date'].dt.is_month_start
    data["is_quarter_end"] = data['date'].dt.is_quarter_end
    data["is_quarter_start"] = data['date'].dt.is_quarter_start
    data["is_year_end"] = data['date'].dt.is_year_end
    data["is_year_start"] = data['date'].dt.is_year_start
    data.drop('date', axis=1, inplace=True)
    return data

data.columns = clean_col_names(data.columns)
data = extract_dates(data)
data = pd.get_dummies(data=data, columns=['holiday', 'seasons', 'functioning_day'])
data.to_parquet(data_interim, compression="snappy")
print("File Cleaned Successfully!")
