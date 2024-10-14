import numpy as np
import pandas as pd
import sklearn as sk
import random

random.seed(2)

df = pd.read_csv("../data/train.csv")

rows = len(df.index)
vars = df.columns.values

'''
8693 observations

['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 
'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 
'VRDeck', 'Name', 'Transported']
'''

sample_size = int(rows*.1) # 869
sample_rows = random.sample(list(df.index), sample_size)

pilot = df.iloc[sample_rows, :]
pilot.to_csv("pilot.csv")