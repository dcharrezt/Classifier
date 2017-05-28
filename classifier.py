import numpy as np
import pandas as pd
import operator
from collections import OrderedDict

def manhattan_distance(x, y):
      return (sum([ abs(a-b) for a, b in zip(x, y) ]))

def mod_standard_score(x):
    x_sorted = np.sort(x)
    card = len(x)
    mean = 0
    if card % 2 == 0:
        mean = (x_sorted[int((card/2)-1)]+x_sorted[int(card/2)]) / 2
    else:
        mean = x_sorted[int(card/2)]
    deviation = sum([ abs(i-mean) for i in x]) / card
    if deviation == 0:
        return [0]*card
    return [(i - mean)/deviation for i in x]

def compare_features(item_1, item_2, features):
    temp = {}
    for x, y, f in zip(item_1, item_2, features):
        if(x!=0 and y!=0):
            temp[abs(x-y)] = f
    ms = OrderedDict(sorted(temp.items(), key=lambda t: t[0]))
    for i in ms:
        print("\t"+str(temp[i])+"\t"+str(i))

def predict():
    features = ['Piano','Voz','Batidas','Blue','Guitar','Coro','Rap','Bpm']
    bands = ['Dog','Phoenix','Heartless','The black','Glee', 'Bad']

    df = pd.read_csv('content_based_music.csv', names = features)
    df.index = bands

    feats = np.array(df.columns.values) # names of the features
    indexes = df.index.values # value of the index
    new_b = []
    band_name = input("band name: ")

    for i in feats:
        new_b.append(float(input("Enter, "+str(i)+":\n")))

    df.loc[band_name] = new_b

    for i in feats:
        temp_v = mod_standard_score(df[i])
        df[i] = temp_v
        # df = df.assign(i=pd.Series(temp_v).values)
    print(df)

    dict_result = {}

    for i in indexes:
        dict_result[i] = manhattan_distance(df.loc[band_name], df.loc[i])

    # print(dict_result)
    r = sorted(dict_result, key=dict_result.__getitem__)

    print("closest to "+str(band_name))
    for i in range(0,3):
        print(r[i]+"\t",dict_result[r[i]])
        compare_features(np.array(df.loc[band_name]), np.array(df.loc[r[i]]), features)

while(True):
    predict()
