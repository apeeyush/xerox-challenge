import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from math import *
from datetime import datetime

# Prepare training dataset by merging csv files
def gen_train_data():
    df_id_age = pd.read_csv('id_age_train.csv')
    df_id_label = pd.read_csv('id_label_train.csv')
    df_id_time_labs = pd.read_csv('id_time_labs_train.csv')
    df_id_time_vitals = pd.read_csv('id_time_vitals_train.csv')
    df_id_age_label_merge = pd.merge(df_id_age, df_id_label, how='inner', on='ID')
    df_id_labs_vitals_merge = pd.merge(df_id_time_labs, df_id_time_vitals, how='inner', on=['ID','TIME'])
    df = pd.merge(df_id_age_label_merge, df_id_labs_vitals_merge, how='inner',on='ID')
    df = df.sort(['ID', 'TIME'])
    return df

# Prepare validation dataset by merging csv files
def gen_val_data():
    df_id_age = pd.read_csv('./validation/id_age_val.csv')
    df_id_time_labs = pd.read_csv('validation/id_time_labs_val.csv')
    df_id_time_vitals = pd.read_csv('validation/id_time_vitals_val.csv')
    df_id_labs_vitals_merge = pd.merge(df_id_time_labs, df_id_time_vitals, how='inner', on=['ID','TIME'])
    df = pd.merge(df_id_age, df_id_labs_vitals_merge, how='inner',on='ID')
    df = df.sort(['ID', 'TIME'])
    return df

def filter_by_label(df, label_val):
    return df[df.LABEL==label_val]

# Filter data to keep only latest timestamp where label=1
def filter_dataset(df):
    label_1 = filter_by_label(df, 1).groupby('ID').tail(2).reset_index()
    label_0 = filter_by_label(df, 0).groupby('ID').tail(15).reset_index()
    final_df = label_1
    final_df = final_df.append(label_0, ignore_index=True)
    return final_df

def transform_gen(df):
    var_list = [
        'V1','V2','V3','V4','V5','V6',
        'L1','L2','L3','L4','L5','L6','L7','L8','L9','L10',
        'L11','L12','L13','L14','L15','L16','L17','L18','L19','L20',
        'L21','L22','L23','L24','L25'
    ]

    # Add running max (maximum of past 10 entries considered)
    for var in var_list:
        col_name = 'max'+var
        df[col_name] = df.groupby('ID')[var].apply(pd.rolling_max, 15, min_periods=1)
    # Add running min (maximum of past 10 entries considered)
    for var in var_list:
        col_name = 'min'+var
        df[col_name] = df.groupby('ID')[var].apply(pd.rolling_min, 15, min_periods=1)
    # Add running mean (maximum of past 10 entries considered)
    for var in var_list:
        col_name = 'mean'+var
        df[col_name] = df.groupby('ID')[var].apply(pd.rolling_mean, 15, min_periods=1)
    return df


def transform_train_data(df):
    # WIP : Add variables to df as needed
    # maxV1 = maxV2 = maxV3 = maxV4 = maxV5 = maxV6 = []
    # minV1 = minV2 = minV3 = minV4 = minV5 = minV6 = []    
    # for index, row in df.iterrows():
    #     pid = row['ID']
    # df['maxV1'] = maxV1
    # df.groupby('ID')['V1'].apply(pd.rolling_mean, 2, min_periods=1)
    df = transform_gen(df)
    # Filter dataset
    df = filter_dataset(df)
    return df

def transform_val_data(df):
    # TODO : Add variables to df as needed
    df = transform_gen(df)
    return df

def drop_columns(df, columns):
    return df.drop(columns, 1)

def predict():
    # Train the model
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rf.fit(X_train_data, y_train_data)

    # Predict using the model
    output_rf = rf.predict(test_data)

    # TODO : Processing the output -> ensuring 0/1, removing negatives etc.

    # Prepare submission
    predictions_file = open("output.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerows(zip(ids, times, output_rf))
    predictions_file.close()

    print 'Done.'


if __name__ == '__main__':
    df = gen_train_data()
    val_df = gen_val_data()

    df = transform_train_data(df)
    val_df = transform_val_data(val_df)


    features = [
        'AGE',

        'L1','L2','L3','L4','L5','L6','L7','L8','L9','L10',
        'L11','L12','L13','L14','L15','L16','L17','L18','L19','L20',
        'L21','L22','L23','L24','L25',

        'V1','V2','V3','V4','V5','V6',

        'maxL1','maxL2','maxL3','maxL4','maxL5','maxL6','maxL7','maxL8','maxL9','maxL10',
        'maxL11','maxL12','maxL13','maxL14','maxL15','maxL16','maxL17','maxL18','maxL19','maxL20',
        'maxL21','maxL22','maxL23','maxL24','maxL25',
        'minL1','minL2','minL3','minL4','minL5','minL6','minL7','minL8','minL9','minL10',
        'minL11','minL12','minL13','minL14','minL15','minL16','minL17','minL18','minL19','minL20',
        'minL21','minL22','minL23','minL24','minL25',
        'meanL1','meanL2','meanL3','meanL4','meanL5','meanL6','meanL7','meanL8','meanL9','meanL10',
        'meanL11','meanL12','meanL13','meanL14','meanL15','meanL16','meanL17','meanL18','meanL19','meanL20',
        'meanL21','meanL22','meanL23','meanL24','meanL25',

        'maxV1','maxV2','maxV3','maxV4','maxV5','maxV6',
        'minV1','minV2','minV3','minV4','minV5','minV6',
        'meanV1','meanV2','meanV3','meanV4','meanV5','meanV6',

        'ICU'
    ]

    # Prepare train data for use by model
    # TODO : Handle any NaN present - Temporarily fill 0
    df = df.fillna(0)
    X_train_data = df[features].values
    y_train_data = df['LABEL'].values

    # Prepare validation data for use by model
    val_df = val_df[val_df.ICU==1]
    # TODO : Handle any NaN present - Temporarily fill 0
    val_df = val_df.fillna(0)
    test_data = val_df[features].values
    ids = val_df['ID'].values
    times = val_df['TIME'].values
    predict()
