import numpy as np
import pandas as pd

df = pd.read_csv("pilot.csv")

def clean_data(data): 
    # Removes unneccessary columns
    data.drop(axis=1, labels=["Unnamed: 0", "PassengerId"])
    return data

def split_cabin_label(cabin_string): 
    # Splits "Cabin" variable into three variables
    # get cabin_string using df.loc[index, "Cabin"]
    cabin_list = cabin_string.split("/")
    # cabin_list[1] = int(cabin_list[1]) # keeping cabin number as a nominal variable
    return cabin_list

df_clean = clean_data(df)
df_pure = df_clean.dropna(axis=0, how="any")
df_pure.to_csv("pilot_pure.csv")