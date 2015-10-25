import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from math import *
from datetime import datetime
import pickle
import sys

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
    if len(sys.argv) > 1:
        id_age_file = sys.argv[1]
        id_time_labs_file = sys.argv[2]
        id_time_vitals_file = sys.argv[3]
    else:
        id_age_file = 'validation/id_age_val.csv'
        id_time_labs_file = 'validation/id_time_labs_val.csv'
        id_time_vitals_file = 'validation/id_time_vitals_val.csv'
    df_id_age = pd.read_csv(id_age_file)
    df_id_time_labs = pd.read_csv(id_time_labs_file)
    df_id_time_vitals = pd.read_csv(id_time_vitals_file)
    df_id_labs_vitals_merge = pd.merge(df_id_time_labs, df_id_time_vitals, how='inner', on=['ID','TIME'])
    df = pd.merge(df_id_age, df_id_labs_vitals_merge, how='inner',on='ID')
    df = df.sort(['ID', 'TIME'])
    return df

def filter_by_label(df, label_val):
    return df[df.LABEL==label_val]

def modify_dead(x):
    timeDeath = x.tail(1)['TIME'].values[0]
    for i, row in x.iterrows():
        if timeDeath - row['TIME'] > 1000:
            x.loc[i,'LABEL'] = 2
    return x

def transform_dead_labels(df):
    df_label_1 = filter_by_label(df,1)
    df_label_1.groupby('ID').apply(modify_dead)
    df_label_1 = filter_by_label(df_label_1,1)
    return df_label_1

# Filter data to keep only latest timestamp where label=1
def filter_dataset(df,i,j):
    df_label_1 = transform_dead_labels(df)
    label_1 = df_label_1.groupby('ID').tail(j).reset_index()
    label_0 = filter_by_label(df, 0).groupby('ID').tail(i).reset_index()
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
        df[col_name] = df.groupby('ID')[var].apply(pd.rolling_max, 50, min_periods=1)
    # Add running min (maximum of past 10 entries considered)
    for var in var_list:
        col_name = 'min'+var
        df[col_name] = df.groupby('ID')[var].apply(pd.rolling_min, 50, min_periods=1)
    # Add running mean (maximum of past 10 entries considered)
    for var in var_list:
        col_name = 'mean'+var
        df[col_name] = df.groupby('ID')[var].apply(pd.rolling_mean, 50, min_periods=1)

    # Add running max (maximum of past 10 entries considered)
    for var in var_list:
        col_name = 'localMax'+var
        df[col_name] = df.groupby('ID')[var].apply(pd.rolling_max, 5, min_periods=1)
    # Add running min (maximum of past 10 entries considered)
    for var in var_list:
        col_name = 'localMin'+var
        df[col_name] = df.groupby('ID')[var].apply(pd.rolling_min, 5, min_periods=1)
    # Add running mean (maximum of past 10 entries considered)
    for var in var_list:
        col_name = 'localMean'+var
        df[col_name] = df.groupby('ID')[var].apply(pd.rolling_mean, 5, min_periods=1)

    return df


def transform_train_data(df,i,j):
    # WIP : Add variables to df as needed
    # maxV1 = maxV2 = maxV3 = maxV4 = maxV5 = maxV6 = []
    # minV1 = minV2 = minV3 = minV4 = minV5 = minV6 = []    
    # for index, row in df.iterrows():
    #     pid = row['ID']
    # df['maxV1'] = maxV1
    # df.groupby('ID')['V1'].apply(pd.rolling_mean, 2, min_periods=1)
    df = transform_gen(df)
    # Filter dataset
    df = filter_dataset(df,i,j)
    return df

def transform_val_data(df,i,j):
    # TODO : Add variables to df as needed
    df = transform_gen(df)
    return df

def drop_columns(df, columns):
    return df.drop(columns, 1)

# class_weight={0:12.5,1:1}

def predict(X_train_data,y_train_data,test_data,ids,times,i,j):
    # Train rf and predict using the model
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    rf.fit(X_train_data, y_train_data)
    output_rf = rf.predict(test_data)
    # Train gbm and predict using the model
    gbm = GradientBoostingClassifier(n_estimators=150, max_depth=5)
    gbm.fit(X_train_data, y_train_data)
    output_gbm = gbm.predict(test_data)

    final_output = [ x||y for x, y in zip(output_rf, output_gbm)]

    # Prepare submission
    predictions_file = open("output_ensemble"+".csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerows(zip(ids, times, final_output))
    predictions_file.close()

    print 'Done.'


def result_gen(i,j):
    df = gen_train_data()
    val_df = gen_val_data()

    df = transform_train_data(df,i,j)
    val_df = transform_val_data(val_df,i,j)

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

        'localMaxL1','localMaxL2','localMaxL3','localMaxL4','localMaxL5','localMaxL6','localMaxL7','localMaxL8','localMaxL9','localMaxL10',
        'localMaxL11','localMaxL12','localMaxL13','localMaxL14','localMaxL15','localMaxL16','localMaxL17','localMaxL18','localMaxL19','localMaxL20',
        'localMaxL21','localMaxL22','localMaxL23','localMaxL24','localMaxL25',
        'localMinL1','localMinL2','localMinL3','localMinL4','localMinL5','localMinL6','localMinL7','localMinL8','localMinL9','localMinL10',
        'localMinL11','localMinL12','localMinL13','localMinL14','localMinL15','localMinL16','localMinL17','localMinL18','localMinL19','localMinL20',
        'localMinL21','localMinL22','localMinL23','localMinL24','localMinL25',
        'localMeanL1','localMeanL2','localMeanL3','localMeanL4','localMeanL5','localMeanL6','localMeanL7','localMeanL8','localMeanL9','localMeanL10',
        'localMeanL11','localMeanL12','localMeanL13','localMeanL14','localMeanL15','localMeanL16','localMeanL17','localMeanL18','localMeanL19','localMeanL20',
        'localMeanL21','localMeanL22','localMeanL23','localMeanL24','localMeanL25',

        'maxV1','maxV2','maxV3','maxV4','maxV5','maxV6',
        'minV1','minV2','minV3','minV4','minV5','minV6',
        'meanV1','meanV2','meanV3','meanV4','meanV5','meanV6',

        'localMaxV1','localMaxV2','localMaxV3','localMaxV4','localMaxV5','localMaxV6',
        'localMinV1','localMinV2','localMinV3','localMinV4','localMinV5','localMinV6',
        'localMeanV1','localMeanV2','localMeanV3','localMeanV4','localMeanV5','localMeanV6',

        'ICU'
    ]

    # Prepare train data for use by model
    df = df[df.ICU == 1]

    train_features = df[features]
    mean = train_features.mean()
    std = train_features.var().apply(np.sqrt)
    train_features = (train_features - mean)/std
    train_features = train_features.fillna(0)
    X_train_data = train_features.values
    y_train_data = df['LABEL'].values

    # Prepare validation data for use by model
    val_df = val_df[val_df.ICU==1]
    test_features = val_df[features]
    test_features = (test_features - mean)/std
    test_features = test_features.fillna(0)
    test_data = test_features.values
    ids = val_df['ID'].values
    times = val_df['TIME'].values
    predict(X_train_data,y_train_data,test_data,ids,times,i,j)


result_gen(220,10)
