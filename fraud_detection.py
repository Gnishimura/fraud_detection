import pandas as pd
import numpy as np
from encoder import Encoder

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.ensemble import (RandomForestClassifier, 
                              GradientBoostingClassifier, 
                              AdaBoostClassifier)
import xgboost as xgb 

ip_data = pd.read_csv('data/IpAddress_to_Country.csv')


def country_from_ip(ip_df, fraud_df):
    """Given a df with lower and upper ranges for ip address, and a df with countries: create a 
    new column in countries df labeled ['ip_address']"""
    v = ip_df.loc[:, 'lower_bound_ip_address':'upper_bound_ip_address'].apply(tuple, 1).tolist()
    idx = pd.IntervalIndex.from_tuples(v, closed='both')
    fraud_df['country'] = ip_data.loc[idx.get_indexer(fraud_df['ip_address'].values), 'country'].values

def clean_df(df):
    """Change to datetime"""
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'], infer_datetime_format=True)
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'], infer_datetime_format=True)
    fraud_data['time_to_purchase'] = fraud_data['purchase_time'] - fraud_data['signup_time']
    fraud_data['days_to_purchase'] = fraud_data['time_to_purchase'].apply(lambda x: x.days)
    clean_fraud_data = fraud_data.drop(['user_id', 'device_id', 
                                        'time_to_purchase', 'signup_time', 
                                        'purchase_time', 'ip_address'], axis=1)

def create_readable_coef_df(coefficients, df):
    features_arr = np.array(df.columns)
    coef_arr = np.array(coefficients).flatten()
    table = np.concatenate([features_arr, coef_arr]).reshape(2, len(df.columns))
    coef_df = pd.DataFrame(table)
    coef_df.columns = coef_df.iloc[0]
    coef_df.drop(index=0, inplace=True)
    return coef_df
