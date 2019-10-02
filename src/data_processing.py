# Developed by: Michail Tzoufras 
# Date updated: 9/23/2019

import os
import csv
import argparse

import pandas as pd
import numpy as np
import copy
#------------------------------------------------------------------

from sklearn import preprocessing
#------------------------------------------------------------------

def normalize_column(df_column, center_at_zero = False):
    """Converts an unnormalized dataframe column to a normalized 
    1D numpy array
    Default: normalizes between [0,1]
    (center_at_zero == True): normalizes between [-1,1] """

    normalized_array = np.array(df_column,dtype = 'float64')
    amax, amin = np.max(normalized_array), np.min(normalized_array)
    normalized_array -= amin
    if center_at_zero:
        normalized_array *= 2.0/(amax-amin)
        normalized_array -= 1.0
    else:
        normalized_array *= 1.0/(amax-amin)
    return normalized_array


def dataframe_to_numpy(df,categorical_columns,ordinal_columns):
    """Converts data in dataframe to numpy array that includes:
    1) one-hot encoded categorical columnhs
    2) normalized ordinal columns"""

    le = preprocessing.LabelEncoder()
    Xtmp = (df[categorical_columns].copy()).apply(lambda col: le.fit_transform(col))

    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore',sparse=False )
    X = np.transpose(ohe.fit_transform(Xtmp))

    for c in ordinal_columns:        
        X = np.vstack([X, normalize_column(df[c])])

    return np.transpose(X)

def combine_rare(df,column, Limit = 200):
    """Combine rare categorical data to Rare_"""
    for r in df[column].unique():
        if (np.sum(df[df[column] ==r]['Status'].value_counts()) < Limit):
            df[column].replace(r,'Rare_'+column,inplace=True)

def oversample_minority(df, ratio = 1.0, random_state=3):
    """Oversamples the minority class to reach a ratio by default
    equal to 1 between the majority and mionority classes"""
    count_class_0, count_class_1 = df['Status'].value_counts()
    df_class_0 = df[df['Status'] == 'paid']
    df_class_1 = df[df['Status'] == 'defaulted']
    #print(count_class_0)
    #print(count_class_1)
    df_class_1_over = df_class_1.sample(int(ratio*count_class_0),replace=True, random_state = random_state)
    df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)
    #print(df_train_over['Status'].value_counts())
    return df_train_over

def undersample_majority(df, ratio = 1.0, random_state=3):
    """Undersamples the majority class to reach a ratio by default
    equal to 1 between the majority and minority classes"""
    count_class_0, count_class_1 = df['Status'].value_counts()
    df_class_0 = df[df['Status'] == 'paid']
    df_class_1 = df[df['Status'] == 'defaulted']
    #print(count_class_0)
    #print(count_class_1)
    df_class_0_under = df_class_0.sample(int(ratio*count_class_1), random_state = random_state)
    df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
    #print(df_train_under['Status'].value_counts)
    return df_train_under
