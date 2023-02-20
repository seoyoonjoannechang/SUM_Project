import sys
[sys.path.append(i) for i in [".", ".."]]

import pandas as pd
from Weather import get_weather
from Pollution import get_pollution
from datetime import datetime
import pickle
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor, StackingClassifier
from xgboost import XGBClassifier

def concat(weather, pollution):
    all_df = pd.concat([weather, pollution], axis=1)
    
    with open("model_features.txt", "r") as f:
        used_cols = []
        for line in f.readlines():
            used_cols.append(line.replace("\n", "").strip())
        
    for col in used_cols:
        if col not in all_df:
            col_df = pd.DataFrame({col:[0]})
            if "lcsCh" in col:
                all_df = pd.concat([all_df, col_df], axis=1)
            if "x0" in col:
                all_df = pd.concat([all_df, col_df], axis=1)
                
    for col in all_df:
        if col not in used_cols:
            all_df.drop(col, axis=1, inplace=True)
            
    return all_df

def load_data():
    weather = get_weather()
    pollution = get_pollution()
    
    df = concat(weather, pollution)

    return fillna(df)


def mean_data():
    with open('pickle/mean.pkl', 'rb') as f:
        mean = pickle.load(f)
    return mean

def fillna(df):
    mean = mean_data()
    mean.set_index("monthtime", inplace=True)

    now_datetime = datetime.now().strftime("%m-%d %H:00:00")

    for col in df.columns:
        if df[col].values.tolist().pop() == "-":
            df[col] = df[col].replace("-", mean.loc[now_datetime, col])

    with open("model_features.txt", "r") as f:
            used_cols = []
            for line in f.readlines():
                used_cols.append(line.replace("\n", "").strip())

    df = df[used_cols]
    df = df.astype(float)

    return df

def model():
    model = joblib.load('./pickle/final_stacking_model.pkl')
    return model

def predict():
    return model().predict(load_data())

def predict_word():
    replace_word = {
        0 : "좋음",
        1 : "보통",
        2 : "나쁨",
        3 : "아주나쁨"
    }
    pred = model().predict(load_data()).tolist().pop()
    return replace_word[pred]