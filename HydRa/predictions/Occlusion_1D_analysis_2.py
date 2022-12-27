#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
from random import shuffle

FAKE_ZEROS=1e-5
## For naive ensembling (probabilistic)
score_df1=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/SONAR+/Classification_cv_scores_run1_SONAR+_final_Ensemble_17_3_7_3_combined10RBPlist_menthaBioPlex_STRING.xls',index_col=0)
score_df2=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/SONAR+/Classification_cv_scores_run2_SONAR+_final_Ensemble_17_3_7_3_combined10RBPlist_menthaBioPlex_STRING.xls',index_col=0)
score_df3=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/SONAR+/Classification_cv_scores_run3_SONAR+_final_Ensemble_17_3_7_3_combined10RBPlist_menthaBioPlex_STRING.xls',index_col=0)
score_df4=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/SONAR+/Classification_cv_scores_run4_SONAR+_final_Ensemble_17_3_7_3_combined10RBPlist_menthaBioPlex_STRING.xls',index_col=0)
score_df5=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/SONAR+/Classification_cv_scores_run5_SONAR+_final_Ensemble_17_3_7_3_combined10RBPlist_menthaBioPlex_STRING.xls',index_col=0)
reference_score_df=pd.concat([score_df1,score_df2,score_df3,score_df4,score_df5])

scores_DNN=list(reference_score_df['DNN_SeqOnly_score'])
scores_SVM=list(reference_score_df['SVM_SeqOnly_score'])
true_labels=list(reference_score_df['RBP_flag'])

def get_TP_FN_TN_FP(scores, true_labels, threshold):
    """pred_labels and true_labels both are array or list of 0 and 1."""
    if(len(scores)!=len(true_labels)):
        raise ValueError('The length of pred_labels and true_labels should be same!')
    
    #print len(true_labels), len(pred_labels)
    TP=len([idx for idx in range(0, len(true_labels)) if true_labels[idx]==1 and scores[idx]>=threshold])
    TN=len([idx for idx in range(0, len(true_labels)) if true_labels[idx]==0 and scores[idx]<threshold])
    FP=len([idx for idx in range(0, len(true_labels)) if true_labels[idx]==0 and scores[idx]>=threshold])
    FN=len([idx for idx in range(0, len(true_labels)) if true_labels[idx]==1 and scores[idx]<threshold])
    return TP, FN, TN, FP

def get_fdr(TP, FN, TN, FP):
    return FP*1.0/(FP+TP+FAKE_ZEROS)

def get_fpr(TP, FN, TN, FP):
    return FP*1.0/(FP+TN+FAKE_ZEROS)

def get_model_fdr(scores,true_labels,threshold):
    TP, FN, TN, FP = get_TP_FN_TN_FP(scores, true_labels, threshold)
    return get_fdr(TP, FN, TN, FP)

def get_ensScore(y_DNN_pred, y_SVM_pred):
	"""
	Calculate ensemble score of DNN's and SVM's scores using probablity ensembling.
	"""
	return 1-get_model_fdr(scores_DNN, true_labels, y_DNN_pred)*get_model_fdr(scores_SVM, true_labels, y_SVM_pred)

#RBPs=['Q14157']
RBPs=['S','N','E','M','nsp10','nsp11','nsp12','nsp13','nsp14','nsp15','nsp16','nsp1','nsp2','nsp3','nsp4','nsp5','nsp6','nsp7','nsp8','nsp9','orf10','orf1ab','orf1a','orf3a','orf3b','orf6','orf7a','orf7b','orf8','orf9b','orf9c']

k=20
out_dir='/home/wjin/data3/Coronavirus/Occlusion_analysis/'

for RBP in RBPs:
    path = os.path.join(out_dir,RBP+'_Occlusion_score_matrix_aac.xls')
    if os.path.exists(path) and (not os.path.exists(path.replace('_aac.xls','_full_aac.xls'))):
        print(RBP)
        df=pd.read_table(path,index_col=0)
        df['occluded_ens_scores']=df.apply(lambda x: get_ensScore(x['occluded_DNN_scores'],x['occluded_SVM_scores']), axis=1)
        df['original_ens_score']=df.apply(lambda x: get_ensScore(x['original_DNN_score'],x['original_SVM_score']), axis=1)
        df['delta_ens']=df.apply(lambda x: x['occluded_ens_scores']-x['original_ens_score'], axis=1)
        df.to_csv(os.path.join(out_dir, RBP+'_Occlusion_score_matrix_full_aac.xls'), index=True, sep='\t')







