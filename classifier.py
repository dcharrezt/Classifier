import numpy as np
import pandas as pd
import operator
import os
from collections import OrderedDict
from sklearn.model_selection import train_test_split

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
    return [(i - mean)/deviation for i in x], mean, deviation

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
        
def cross_validation(dataset):
	path = "datasets/"+dataset+"_/"

	filelist = os.listdir(path)
		
	class_count = {}
	class_count['Gymnastics'] = [0,0,0]
	class_count['Basketball'] = [0,0,0]
	class_count['Track'] = [0,0,0]
	
	class_t = {}
	class_t['Gymnastics'] = 0
	class_t['Basketball'] = 0
	class_t['Track'] = 0
	
	for test in range(0,10):
		frame = pd.DataFrame()
		list_ = []
		print("Test ", filelist[test])
		for i in range(0,10):
			if test != i:
				df = pd.read_csv(path+filelist[i],index_col=0, header=0,sep='\t')
				list_.append(df)
		frame = pd.concat(list_)
		
		frame['num'], m, d = mod_standard_score(np.array(frame['num']))
		frame['num.1'], m1, d1 = mod_standard_score(np.array(frame['num.1']))
		
		df2 = pd.read_csv(path+filelist[test],index_col=0, header=0,sep='\t')

		for index, row in df2.iterrows():
	
			min_ = 99999
			test_tmp = [(row['num']-m)/d, (row['num.1']-m1)/d1]
			for index_, row_ in frame.iterrows():

				tmp = [row_['num'], row_['num.1']]
				dist = manhattan_distance(tmp, test_tmp)
#				print (row_['class'], dist)
				if(dist < min_):
					min_ = dist
#					print("test ", test_tmp)
#					print("tmp ", tmp)
					res = [row_['class'], dist]
					
			class_t[row['class']] += 1

			if row['class'] == 'Gymnastics':
				if res[0] == 'Gymnastics':
					class_count['Gymnastics'][0] += 1
				if res[0] == 'Basketball':
					class_count['Gymnastics'][1] += 1
				if res[0] == 'Track':
					class_count['Gymnastics'][2] += 1
			if row['class'] == 'Basketball':
				if res[0] == 'Gymnastics':
					class_count['Basketball'][0] += 1
				if res[0] == 'Basketball':
					class_count['Basketball'][1] += 1
				if res[0] == 'Track':
					class_count['Basketball'][2] += 1
			if row['class'] == 'Track':
				if res[0] == 'Gymnastics':
					class_count['Track'][0] += 1
				if res[0] == 'Basketball':
					class_count['Track'][1] += 1
				if res[0] == 'Track':
					class_count['Track'][2] += 1
			
#			print (res)

	test_sum = 0
	c = 0
	col_sum = [0,0,0]
	for i in class_count:
		print(i + '\t\t' + str(class_count[i]))
		test_sum += class_count[i][c]
		col_sum[0] +=  class_count[i][0]
		col_sum[1] +=  class_count[i][1]
		col_sum[2] +=  class_count[i][2]
		c+=1
		
		
	total_sum = 0
	for i in class_t:
		total_sum += class_t[i]

	ac = test_sum / total_sum
	print("Accuracy: ", ac)
#	print (class_t)
	
	kappa_count = {}
	kappa_count['Gymnastics'] = [0,0,0]
	kappa_count['Basketball'] = [0,0,0]
	kappa_count['Track'] = [0,0,0]

	
	for i in kappa_count:
		kappa_count[i][0] = round(( col_sum[0]/total_sum ) * class_t[i])
		kappa_count[i][1] = round(( col_sum[1]/total_sum ) * class_t[i])
		kappa_count[i][2] = round(( col_sum[2]/total_sum ) * class_t[i])
	
	kappa_col = [0,0,0]
	ms = 0
	kappa_sum = 0
	for i in kappa_count:
		print(i + '\t\t' + str(kappa_count[i]))
		kappa_sum += kappa_count[i][ms]
		kappa_col[0] +=  kappa_count[i][0]
		kappa_col[1] +=  kappa_count[i][1]
		kappa_col[2] +=  kappa_count[i][2]
		ms+=1
		
	acc = kappa_sum / sum(kappa_col)
	
	print("Kappa Score: ", (ac - acc) / (1 - acc) )



cross_validation("athletes")





