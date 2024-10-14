# -*- coding: utf-8 -*-
"""
Cleaning Pipeline

Functions for cleaning/preparing the dataset
"""
import numpy as np
import pandas as pd

import sklearn.tree as sktree
import sklearn.preprocessing as preprocessing

def split_cabin_label(cabin_string): 
    # Splits "Cabin" variable
    # get cabin_string using df['Cabin'].iloc[index]
    if type(cabin_string) == float:
        return float('NaN'), float('Nan') # catching missing data
    else:
        cabin_list = cabin_string.split("/")
        cabin_deck = cabin_list[0]
        cabin_side = cabin_list[2]
        return cabin_deck, cabin_side
    
def create_cabin_variables(df):
    cabin_deck_list, cabin_side_list = [], []
    for i in range(len(df.index)):
        cabin_val = df['Cabin'].iloc[i]
        cabin_deck, cabin_side = split_cabin_label(cabin_val)
        cabin_deck_list.append(cabin_deck)
        cabin_side_list.append(cabin_side)
    
    df_clean = df.copy(deep=True)
    df_clean.drop(columns='Cabin', inplace=True)
    df_clean.insert(4,'CabinDeck', cabin_deck_list)
    df_clean.insert(5, 'CabinSide', cabin_side_list)
    return df_clean

def encode_data(df):
    xvars = list(df.columns[:-1])

    encoder = preprocessing.OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df[xvars].values)
    encoded_features = encoder.get_feature_names_out(input_features=xvars)
    df_res = pd.DataFrame(encoded_data, columns=encoded_features)
    df_res['Transported'] = df['Transported'].values.astype(int)
    return df_res

# Testing first pipeline
# pilot = pd.read_csv("pilot.csv")
# vars_of_interest = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP','Transported'] 
# df = pilot[vars_of_interest]

# df_clean =  create_cabin_variables(df)
# df_res= encode_data(df_clean)
# print(df_res.head())
