import numpy as np
import pandas as pd
import csv
import sys
from os import listdir
from os.path import isfile, join

def test_train_split(df):
	id_list = df['ID'].unique()
	np.random.shuffle(id_list)
	size = len(id_list)
	train_size = int(size*0.8)
	test_size = size - train_size
	train_data = df[df['ID'].isin(id_list[0:train_size])]
	test_data = df[df['ID'].isin(id_list[train_size:size])]
	return train_data,test_data

def find_labels(df):
	groups = df.groupby('ID')
	labels = []
	ids = []
	time_pred = []
	for name,g in groups:
		maxv = 0
		time_p = 0
		ids += [name]
		for index,row in g.iterrows():
			if row['LABEL'] == 1:
				maxv = 1
				time_p = row['TIME']
				break
		labels += [maxv]
		time_pred += [g.tail(1)['TIME'].values[0] - time_p]
	new_df = pd.DataFrame({'ID':ids,'TIME':time_pred,'LABEL':labels})
	return new_df

def scores(result,truth):
	combined_truth_result = pd.merge(result,truth,how='inner',on='ID')
	# print combined_truth_result
	Pos = combined_truth_result[combined_truth_result['LABEL_x'] == 1]
	# print Pos
	Neg = combined_truth_result[combined_truth_result['LABEL_x'] == 0]
	TP = (Pos['LABEL_x'] == Pos['LABEL_y']).sum()
	FP = (Pos['LABEL_x'] != Pos['LABEL_y']).sum()
	TN = (Neg['LABEL_x'] == Neg['LABEL_y']).sum()
	FN = (Neg['LABEL_x'] != Neg['LABEL_y']).sum()
	Specificity = TN*1.0/(TN+FP)
	Sensitivity = TP*1.0/(TP+FN)
	time = Pos[Pos['LABEL_x'] == Pos['LABEL_y']]['TIME'].median()*1.0/3600.0
	Score = 0.75*Sensitivity + 0.2*(Specificity-0.99)
	return Score, Specificity, Sensitivity,time

def compute_score(output_file):
	truth = pd.read_csv('validation/id_label_val.csv')
	truth.columns = ['ID','LABEL']
	result = pd.read_csv(output_file)
	result.columns = ['ID','TIME','LABEL']
	return scores(find_labels(result),truth)

# Usage :
# combine_result('output_poss_110_2.csv','output_poss_125_2.csv','out_110_2_125_2.csv')
def combine_result(f1,f2,out_name):
	r1 = pd.read_csv(f1)
	r2 = pd.read_csv(f2)
	r1.columns = ['ID','TIME','LABEL']
	r2.columns = ['ID','TIME','LABEL']
	# print r1
	# print r2
	r3 = (np.logical_and((r1['LABEL'] == 1).values,(r2['LABEL'] == 1).values)).astype(int)
	predictions_file = open("ensemble_out/"+out_name, "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerows(zip(r1['ID'].values, r1['TIME'].values, r3))
	predictions_file.close()

def create_output_from_prob():
	pass

# Evaluate submission
sc,sp,sens,time = compute_score('output_rf.csv')
time  = time if time < 72.0 else time
sc = sc + 0.05*(72-time)/72
print('Score = %.4f, Specificity = %.4f, Sensitivity = %.4f, Prediction Time = %.4f\n'%(sc,sp,sens,time))

# Evaluate all files in a folder
# for f in listdir('ICU_out'):
# 	if (isfile(join('ICU_out/',f)) and (f.find('output_poss_better_800_') > -1)):
#  		print f
#  		sc,sp,sens,time = compute_score('ICU_out/'+f)
#  		time  = time if time < 72.0 else time
#  		sc = sc + 0.05*(72-time)/72
#  		print('Score = %.4f, Specificity = %.4f, Sensitivity = %.4f, Prediction Time = %.4f\n'%(sc,sp,sens,time))
# # Run validatio for a single file
# sc,sp,sens,time = compute_score('ensemble_out/out_110_2_125_2.csv')
# time  = time if time < 72.0 else time
# sc = sc + 0.05*(72-time)/72
# print('Score = %.4f, Specificity = %.4f, Sensitivity = %.4f, Prediction Time = %.4f\n'%(sc,sp,sens,time))

#l1 = []
#for f in listdir('.'):
#	if (isfile(join('.',f)) and (f.find('output_poss_better_2000_') > -1)):
#		l1 += [f]

#l2 = []
#for f in listdir('.'):
#	if (isfile(join('./',f)) and (f.find('out_g') > -1)):
#		l2 += [f]

#for i in f1:
#	for j in f2:
#		print f1,f2
#		t = f1.split('.')[0]+f2.split('.')[0]+'.csv'
#		combine_result(f1,f2,t)
#		sc,sp,sens,time = compute_score('ensemble_out/'+t)
#		time  = time if time < 72.0 else time
#		sc = sc + 0.05*(72-time)/72
#		print('Score = %.4f, Specificity = %.4f, Sensitivity = %.4f, Prediction Time = %.4f\n'%(sc,sp,sens,time))
#		print >> sys.stderr, 'Score = %.4f, Specificity = %.4f, Sensitivity = %.4f, Prediction Time = %.4f\n'%(sc,sp,sens,time)

# print "out_110_2_125_2.csv"

# # sc,sp,sens,time = compute_score('output_poss_138_2.csv')
# # time  = time if time < 72.0 else time
# # sc = sc + 0.05*(72-time)/72
# # print('Score = %.4f, Specificity = %.4f, Sensitivity = %.4f, Prediction Time = %.4f\n'%(sc,sp,sens,time))


# sc,sp,sens,time = compute_score('ICU_out/output_poss_133_2.csv')
# time  = time if time < 72.0 else time
# sc = sc + 0.05*(72-time)/72
# print('Score = %.4f, Specificity = %.4f, Sensitivity = %.4f, Prediction Time = %.4f\n'%(sc,sp,sens,time))
