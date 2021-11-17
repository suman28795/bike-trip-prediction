import os
dir_path = os.getcwd()
print("DIR PATH" + dir_path)
static_dir_path = os.path.join(dir_path, "Static")
report_path = os.path.join(static_dir_path, "For_modeling.csv")
print(report_path)

import pandas as pd
seoul_data = pd.read_csv(report_path, index_col=[0])
seoul_data.drop(labels=["Snow", "Precip", "PLatd", "PLong", "DLatd", "DLong"], axis=1, inplace=True)
seoul_data = seoul_data.loc[seoul_data["Dust"] * seoul_data["Wind"] * seoul_data["Haversine"] * seoul_data["Solar"]!= 0.0]
seoul_data.reset_index(drop=True, inplace=True)
seoul_data.drop(labels=["Dday", "Dmonth", "DDweek"], axis=1, inplace=True)

#FEATURE ENGINEERING
seoul_data_sample = seoul_data.sample(n=53609, replace=True,random_state=101)
seoul_data_sample.reset_index(drop=True,inplace=True)
X = seoul_data_sample.drop(labels=["Duration"], axis = 1)
y = seoul_data_sample["Duration"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train,y_train)
X_test_rfe = rfe.transform(X_test)
feature_list_rfe = [col for i,col in enumerate(X_train.columns) if rfe.support_[i]]

#MODELING
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1)
rf.fit(X_train_rfe, y_train)
y_hat_test = rf.predict(X_test_rfe)
y_hat_train = rf.predict(X_train_rfe)

from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
print(f'Training score : {rf.score(X_train_rfe, y_train)}')
print('r2 score:', r2_score(y_test, y_hat_test))
print('MAE:', mean_absolute_error(y_test, y_hat_test))
print('MSE:', mean_squared_error(y_test, y_hat_test))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_hat_test)))

from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train_rfe, y_train)
y_hat_test = xgb.predict(X_test_rfe)

import joblib
joblib.dump(xgb, 'final_model_best.joblib')
with open("../final_model_best.joblib", 'wb') as x:
    joblib.dump(xgb, x, compress=3)

xgb_final = joblib.load(dir_path + "\\final_model_best.joblib")
print(xgb_final)

import numpy as np
def predict_duration(attributes: np.ndarray):
    pred = xgb_final.predict(attributes)
    print("Duration Predicted")
    return int(pred[0])


