import numpy as np
import pandas as pd

df = pd.read_csv("pilot.csv")

df.drop(columns=["Unnamed: 0", "PassengerId"], inplace=True)

vars_of_interest = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP','Transported'] 
df_test = df[vars_of_interest]

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
    
cd_list, cs_list= [], []
for i in range(len(df_test.index)):
    g = df_test['Cabin'].iloc[i]
    x, y = split_cabin_label(g)
    cd_list.append(x)
    cs_list.append(y)

df_test.drop(columns='Cabin', inplace=True)
df_test.insert(4,'CabinDeck', cd_list)
df_test.insert(5, 'CabinSide', cs_list)

df_test.to_csv("pilot_test.csv")

df_test.fillna(value='_MissingValue', inplace=True)

z = pd.crosstab(df_test['Transported'], df_test['HomePlanet'])

for i, v in enumerate(df_test.columns):
    if v == 'Transported':
        break
    else:
        contingency_table = pd.crosstab(df_test['Transported'], df_test[v])
        fname = "contingency_table_%s.csv" % v
        contingency_table.to_csv(fname)
