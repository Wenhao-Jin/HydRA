#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFECV, SelectFromModel, chi2, f_classif, VarianceThreshold, mutual_info_classif, SelectFdr
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LassoCV
from random import shuffle
from sklearn.metrics import roc_curve, auc, recall_score, precision_recall_curve, average_precision_score
# from scipy import interp, stats
import math
# from skfeature.function.information_theoretical_based import MIFS,MRMR,CIFE,JMI
# from skfeature.function.sparse_learning_based import RFS
import multiprocessing as mp

plt.switch_backend('agg')


# Info_df = pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/software/SONAR/menthaBioPlexSTRING_feature_table_combined8RBPlist.xls', index_col=0)
# prots=list(filter(lambda x: os.path.exists('/home/wjin/data2/proteins/uniport_data/canonical_seq/'+x+'.spd3'),Info_df.index))
# Info_df = Info_df.loc[prots]
# Info_df.RBP_flag=Info_df.RBP_flag.apply(lambda x: 1 if x else 0)

def chi2_filtering_followedBy_SVMselection(mer, out_dir, alpha=0.01, model_name='SVM_kmer3_chi2_alpha'):
	selector = SelectFdr(score_func=chi2, alpha=alpha)
	selector.fit(np.array(mer.iloc[:,:-1]),np.array(mer.iloc[:,-1]))
	if sum(selector.get_support())==0:
		f=open(os.path.join(out_dir, model_name+'_selected_features_From_WholeDataSet.txt'),'w')
		f.write('')
		f.close()
	else:
		Xy_mer_df=mer[list(mer.columns[:-1][selector.get_support()])+['RBP_flag']]
		clf0=LinearSVC()
		sfm = SelectFromModel(clf0)
		#print(Xy_mer_df.iloc[:,:-1])
		#print(Xy_mer_df.iloc[:,-1])
		sfm.fit(Xy_mer_df.iloc[:,:-1], Xy_mer_df.iloc[:,-1])
		Xy_df=Xy_mer_df[list(Xy_mer_df.columns[:-1][sfm.get_support()])+['RBP_flag']]
		f=open(os.path.join(out_dir, model_name+'_selected_features_From_WholeDataSet.txt'),'w')
		f.write('\n'.join(list(Xy_mer_df.columns[:-1][sfm.get_support()])))
		f.close()

    #get_scores_for_allprot_via_cvTesting(Xy_df, 'RBP_flag', model_name+str(alpha))

def chi2_filtering_Only(mer, out_dir, alpha=0.01, model_name='SVM_kmer3_chi2_alpha'):
	selector = SelectFdr(score_func=chi2, alpha=alpha)
	print(mer.iloc[:,:-1])
	selector.fit(np.array(mer.iloc[:,:-1]),np.array(mer.iloc[:,-1]))
	#Xy_mer_df=mer[list(mer.columns[:-1][selector.get_support()])+['RBP_flag']]
	f=open(os.path.join(out_dir,model_name+'_selected_features_From_WholeDataSet.txt'),'w')
	if sum(selector.get_support())==0:
		f.write('')
	else:
		f.write('\n'.join(list(mer.columns[:-1][selector.get_support()])))
	f.close()

if __name__=='__main__':
	mer3=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/kmer/K3_mers1000_feature_table_combined8RBPlist_menthaBioPlexSTRING.txt',index_col=0).drop('RBP_flag',axis=1)
	mer4=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/kmer/K4_mers1000_feature_table_combined8RBPlist_menthaBioPlexSTRING.txt',index_col=0).drop('RBP_flag',axis=1)
	#mer5=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/kmer/K5_mers1000_feature_table_combined8RBPlist_menthaBioPlexSTRING.txt',index_col=0).drop('RBP_flag',axis=1)
	#mer6=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/kmer/K6_mers1000_feature_table_combined8RBPlist_menthaBioPlexSTRING.txt',index_col=0).drop('RBP_flag',axis=1)
	#mer7=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/kmer/K7_mers1000_feature_table_combined8RBPlist_menthaBioPlexSTRING.txt',index_col=0).drop('RBP_flag',axis=1)

	mer3=mer3.join(Info_df.RBP_flag,how='inner')
	mer4=mer4.join(Info_df.RBP_flag,how='inner')
	#mer5=mer5.join(Info_df.RBP_flag,how='inner')
	#mer6=mer6.join(Info_df.RBP_flag,how='inner')
	#mer7=mer7.join(Info_df.RBP_flag,how='inner')

	#SS_10mer=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/SS_kmer/Secondary_stucture_K10_mers1500_feature_table_combined8RBPlist_addMissingProteins.txt',index_col=0).drop('RBP_flag',axis=1)
	SS_11mer=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/SS_kmer/Secondary_stucture_K11_mers1500_feature_table_combined8RBPlist_addMissingProteins.txt',index_col=0).drop('RBP_flag',axis=1)
	SS_15mer=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/SS_kmer/Secondary_stucture_K15_mers1500_feature_table_combined8RBPlist_addMissingProteins.txt',index_col=0).drop('RBP_flag',axis=1)
	#SS_19mer=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/SS_kmer/Secondary_stucture_K19_mers1500_feature_table_combined8RBPlist_addMissingProteins.txt',index_col=0).drop('RBP_flag',axis=1)
	#SS_23mer=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/SS_kmer/Secondary_stucture_K23_mers1500_feature_table_combined8RBPlist_addMissingProteins.txt',index_col=0).drop('RBP_flag',axis=1)

	#SS_10mer=SS_10mer.join(Info_df.RBP_flag,how='inner')
	SS_11mer=SS_11mer.join(Info_df.RBP_flag,how='inner')
	SS_15mer=SS_15mer.join(Info_df.RBP_flag,how='inner')
	#SS_19mer=SS_19mer.join(Info_df.RBP_flag,how='inner')
	#SS_23mer=SS_23mer.join(Info_df.RBP_flag,how='inner')

	PseAAC=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/PseAAC_feature_table_menthaBioPlexSTRING.txt',index_col=0)
	PseAAC=PseAAC.join(Info_df.RBP_flag,how='inner')

	pool=mp.Pool(processes=4)
	alpha=0.01
	out_dir='/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/'
	results=[pool.apply(chi2_filtering_followedBy_SVMselection, args=(mer,out_dir,alpha,'SVM_'+name+'_chi2_alpha'+str(alpha)+'_with_LinearSVC',)) for mer, name in zip([mer3,mer4,SS_11mer,SS_15mer],['mer3','mer4','SS_11mer','SS_15mer'])]

	chi2_filtering_Only(PseAAC,out_dir,alpha,'SVM_PseAAC_chi2_alpha'+str(alpha))

	f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_mer3_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
	mer3_selected=f.read().split('\n')
	f.close()
	f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_mer4_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
	mer4_selected=f.read().split('\n')
	f.close()
	f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_SS_11mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
	SSmer11_selected=f.read().split('\n')
	f.close()
	f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_SS_15mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
	SSmer15_selected=f.read().split('\n')
	f.close()
	f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_PseAAC_chi2_alpha0.01_selected_features_From_WholeDataSet.txt')
	PseAAC_selected=f.read().split('\n')
	f.close()


	union_df=mer3[mer3_selected].join(mer4[mer4_selected],how='inner').join(SS_11mer[SSmer11_selected],how='inner').join(SS_15mer[SSmer15_selected],how='inner').join(PseAAC[PseAAC_selected],how='inner')
	union_df=union_df.join(Info_df.RBP_flag,how='inner')
	union_df.to_csv('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_SeqFeature_table_comined_all_selected_features_From_WholeDataset_12Apr.txt',index=True,sep='\t')




