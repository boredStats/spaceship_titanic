import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sklearn.tree as sktree

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

'''Colormap testing lol
# cmap = sns.cm.rocket_r
# cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
# cmap = sns.light_palette("seagreen", as_cmap=True)
'''

'''Heatmaps
cmap = sns.color_palette("YlOrBr", as_cmap=True)

for v in df_test.columns:
    if v == 'Transported':
        break
    else:
        contingency_table = pd.crosstab(df_test['Transported'], df_test[v])
        contingency_table.to_csv("contingency_table_%s.csv" % v)
        ax = sns.heatmap(contingency_table, annot=True, fmt='d', cmap=cmap)
        fig = ax.get_figure()
        fig.savefig("heatmap_%s.png" % v)
        plt.clf()

# pandas crosstab does not include NaN as a category, code below is for considering NaN as a variable level

df_test.fillna(value='_MissingValue', inplace=True) # replacing NaN with string for later

c_tables = []
for v in df_test.columns:
    if v == 'Transported':
        break
    else:
        contingency_table = pd.crosstab(df_test['Transported'], df_test[v])
        c_tables.append(contingency_table)
        fname = "contingency_table_%s.csv" % v
        contingency_table.to_csv(fname)

test = c_tables[2]

sns.heatmap(test,annot=True, fmt='d')
plt.show()
'''

xvars = list(df_test.columns[:-1])
y = ['Transported']

df2 = df_test.copy(deep=True)

def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


for x in xvars:
    #a, _ = encode_target(df2, x)
    targets = df2[x].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df2[x] = df2[x].replace(map_to_int)

    #print(a.head())
    
clf = sktree.DecisionTreeClassifier(random_state=2)
clf.fit(df2[xvars], df_test[y])
sktree.plot_tree(clf)
plt.show()

