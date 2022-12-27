#!/usr/bin/env python
## v2: simplified the computation of SVM k-mer/SSkmer count updates.


import pandas as pd
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, recall_score, precision_recall_curve, average_precision_score
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense, Input, Dropout, Activation, merge, Layer, InputSpec, add, Concatenate
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization
from keras import metrics
import keras.backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy import interp, stats
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Reshape
from keras import regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import clone 
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing as mp
from random import shuffle
import joblib
from pathlib import Path
from ..models.Sequence_class import Protein_Sequence_Input5, Protein_Sequence_Input5_2, Protein_Sequence_Input5_2_noSS
from ..models.DNN_seq import SONARp_DNN_SeqOnly, SONARp_DNN_SeqOnly_noSS
from ..preprocessing.get_SVM_seq_features import Get_feature_table, Get_feature_table_noSS

import pkg_resources
import warnings
from pathlib import Path
import warnings

plt.switch_backend('agg')

FAKE_ZEROS=1e-5


def oversampling(X, y):
    """
    PS: The last column of Xy_df should be the ycol
    """
    Positive_X=X[np.nonzero(y)[0]]
    Negative_X=X[np.where(y==0)[0]]
    folds=int(round(len(Negative_X)*1.0/len(Positive_X)))
    Positive_X_oversampled=np.concatenate([Positive_X]*folds)
    X_new=np.concatenate([Negative_X,Positive_X_oversampled])
    y_new=np.concatenate([np.zeros(len(Negative_X)), np.ones(len(Positive_X_oversampled))])
    return X_new, y_new

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

def get_ensScore(y_DNN_pred, y_SVM_pred, scores_DNN=None, scores_SVM=None, true_labels=None):
    """
    Calculate ensemble score of DNN's and SVM's scores using probablity ensembling.
    """
    return 1-get_model_fdr(scores_DNN, true_labels, y_DNN_pred)*get_model_fdr(scores_SVM, true_labels, y_SVM_pred)

def merge_continuous_range(coord_list):
    """
    coord: coordinate sets, a nested tuple, contains: ((row_index_start, row_index_stop), (column_index_start, column_index_end)) 
    """
    ##Sort row coordinates, first "start", second "end"
    coord_list.sort(key=lambda x:(x[0],x[1]))
    merged_coord=[]
    if len(coord_list)==0:
        return []

    tmp_coord=coord_list[0]
    for i in range(1, len(coord_list)):
        if coord_list[i][0]>tmp_coord[1]+1:
            merged_coord.append(tmp_coord)
            tmp_coord=coord_list[i]
        else:
            tmp_coord=(min(tmp_coord[0],coord_list[i][0]), max(tmp_coord[0],coord_list[i][1]))
    
    merged_coord.append(tmp_coord)
    
    return merged_coord

def up_count_kmer(RBP_uniprotID, seq, start, end, k_mer_l, tmp_feature_table, mer_selected):
    """
    seq: string, protein sequence or SS sequence
    i: int, the start index of occluder
    k: int, the length of occluder.
    k_mer_l: int, length of k-mer
    """
    for j in range(max(0, start), min(end, len(seq))-(k_mer_l-1)):
        if seq[j:j+k_mer_l] in mer_selected:
            tmp_feature_table[seq[j:j+k_mer_l]]+=1

def up_count_kmer_2(kmer, tmp_feature_table, mer_selected):
    """
    Simply add 1 to the kmer's corresponding feature value.
    """
    if kmer in mer_selected:
        tmp_feature_table[kmer]+=1
    
def down_count_kmer(RBP_uniprotID, seq, start, end, k_mer_l, tmp_feature_table, mer_selected):
    """
    seq: string, protein sequence or SS sequence
    i: int, the start index of occluder
    k: int, the length of occluder.
    k_mer_l: int, length of k-mer
    """
    for j in range(max(0, start), min(end, len(seq))-(k_mer_l-1)):
        if seq[j:j+k_mer_l] in mer_selected:
            tmp_feature_table[seq[j:j+k_mer_l]]-=1
            if tmp_feature_table[seq[j:j+k_mer_l]]<0:
                raise ValueError("Feature ("+seq[j:j+k_mer_l]+") counting problems in "+RBP_uniprotID+".")

def down_count_kmer_2(kmer, tmp_feature_table, mer_selected):
    """
    Simply subtract 1 to the kmer's corresponding feature value.
    """
    if kmer in mer_selected:
        tmp_feature_table[kmer]-=1
        if tmp_feature_table[kmer]<0:
                raise ValueError("Feature ("+seq[j:j+k_mer_l]+") counting problems in "+RBP_uniprotID+".")


def recount_kmer(RBP_uniprotID, seq, i, k, k_mer_l, tmp_feature_table, mer_selected):
    """
    seq: string, protein sequence or SS sequence
    i: int, the start index of occluder
    k: int, the length of occluder.
    k_mer_l: int, length of k-mer
    """
    for j in range(max(0, i-(k_mer_l-1)), min(i+k+(k_mer_l-1), len(seq))-(k_mer_l-1)):
        if seq[j:j+k_mer_l] in mer_selected:
            tmp_feature_table[seq[j:j+k_mer_l]]-=1
            if tmp_feature_table[seq[j:j+k_mer_l]]<0:
                raise ValueError("Feature ("+seq[j:j+k_mer_l]+") counting problems in "+RBP_uniprotID+".")

def update_kmer_count_2D(RBP_uniprotID, seq, i1, i2, k, k_mer_l, tmp_feature_table, mer_selected):
    """
    seq: string, protein sequence or SS sequence
    i1: int, the start index of 1st occluder
    i2: int, the start index of 2nd occluder
    k: int, the length of occluder.
    k_mer_l: int, length of k-mer
    """
    if abs(i1-i2)>=k+k_mer_l-1:
        # recount_kmer(RBP_uniprotID, seq, i1, k, 3, tmp_feature_table)
        # recount_kmer(RBP_uniprotID, seq, i2, k, 3, tmp_feature_table)
        down_count_kmer(RBP_uniprotID, seq, i1-(k_mer_l-1), i1+k+(k_mer_l-1), k_mer_l, tmp_feature_table, mer_selected)  
        down_count_kmer(RBP_uniprotID, seq, i2-(k_mer_l-1), i2+k+(k_mer_l-1), k_mer_l, tmp_feature_table, mer_selected)  
    else:
        tmp_start=max(i1-(k_mer_l-1),i2-(k_mer_l-1))
        tmp_end=min(i1+k+(k_mer_l-1),i2+k+(k_mer_l-1))
        up_count_kmer(RBP_uniprotID, seq, start, end, k_mer_l, tmp_feature_table, mer_selected)
        down_count_kmer(RBP_uniprotID, seq, i1-(k_mer_l-1), i1+k+(k_mer_l-1), k_mer_l, tmp_feature_table, mer_selected)  
        down_count_kmer(RBP_uniprotID, seq, i2-(k_mer_l-1), i2+k+(k_mer_l-1), k_mer_l, tmp_feature_table, mer_selected)  


def get_selected_SVM_features_1D_occlusion(RBP_uniprotID, i, k, seq, ss, svm_feature_table=None, selected_aa3mers=None, selected_aa4mers=None, selected_SS11mers=None, selected_SS15mers=None, selected_AAC=None, AAs=None):
    """
    i: int, the start index of occluder
    k: int, the length of occluder.
    """
    #### Prepare occlusion for SVM
    tmp_feature_table=svm_feature_table.loc[RBP_uniprotID]
    
    ### decrease corresponding k-mer
    ## 3-mer
    #recount_kmer(RBP_uniprotID, seq, i, k, 3, tmp_feature_table)
    down_count_kmer(RBP_uniprotID, seq, i-(3-1), i+k+(3-1), 3, tmp_feature_table, selected_aa3mers)   
    ## 4-mer
    #recount_kmer(RBP_uniprotID, seq, i, k, 4, tmp_feature_table)
    down_count_kmer(RBP_uniprotID, seq, i-(4-1), i+k+(4-1), 4, tmp_feature_table, selected_aa4mers)   
    ### decrease corresponding SS-kmer
    ## 11-mer
    #recount_kmer(RBP_uniprotID, ss, i, k, 11, tmp_feature_table)
    down_count_kmer(RBP_uniprotID, ss, i-(11-1), i+k+(11-1), 11, tmp_feature_table, selected_SS11mers)   

    ## 15-mer
    #recount_kmer(RBP_uniprotID, ss, i, k, 15, tmp_feature_table)
    down_count_kmer(RBP_uniprotID, ss, i-(15-1), i+k+(15-1), 15, tmp_feature_table, selected_SS15mers)   

    ### re-run PseAAC
    seq_occ=seq[:i]+seq[(i+k):]
    tmp_paac=get_AAC_features(seq_occ, AAs) ## the output is a dictionary
    if not tmp_paac:
        raise ValueError(RBP_uniprotID+"Sequence ("+seq_occ+") gets problem in calculating AAC.")
    else:    
        for p in selected_AAC:
            tmp_feature_table[p]=tmp_paac[p]

    return tmp_feature_table

def get_selected_SVM_features_noSS_1D_occlusion(RBP_uniprotID, i, k, seq, svm_feature_table=None, selected_aa3mers=None, selected_aa4mers=None, selected_AAC=None, AAs=None):
    """
    i: int, the start index of occluder
    k: int, the length of occluder.
    """
    #### Prepare occlusion for SVM
    tmp_feature_table=svm_feature_table.loc[RBP_uniprotID]
    
    ### decrease corresponding k-mer
    ## 3-mer
    #recount_kmer(RBP_uniprotID, seq, i, k, 3, tmp_feature_table)
    down_count_kmer(RBP_uniprotID, seq, i-(3-1), i+k+(3-1), 3, tmp_feature_table, selected_aa3mers)   
    ## 4-mer
    #recount_kmer(RBP_uniprotID, seq, i, k, 4, tmp_feature_table)
    down_count_kmer(RBP_uniprotID, seq, i-(4-1), i+k+(4-1), 4, tmp_feature_table, selected_aa4mers)   
    ### decrease corresponding SS-kmer
    # ## 11-mer
    # #recount_kmer(RBP_uniprotID, ss, i, k, 11, tmp_feature_table)
    # down_count_kmer(RBP_uniprotID, ss, i-(11-1), i+k+(11-1), 11, tmp_feature_table, SSmer11_selected)   

    # ## 15-mer
    # #recount_kmer(RBP_uniprotID, ss, i, k, 15, tmp_feature_table)
    # down_count_kmer(RBP_uniprotID, ss, i-(15-1), i+k+(15-1), 15, tmp_feature_table, SSmer15_selected)   

    ### re-run PseAAC
    seq_occ=seq[:i]+seq[(i+k):]
    tmp_paac=get_AAC_features(seq_occ, AAs) ## the output is a dictionary
    if not tmp_paac:
        raise ValueError(RBP_uniprotID+"Sequence ("+seq_occ+") gets problem in calculating AAC.")
    else:    
        for p in selected_AAC:
            tmp_feature_table[p]=tmp_paac[p]

    return tmp_feature_table

def get_selected_SVM_features_2D_occlusion(RBP_uniprotID, i1, i2, k, seq, ss, svm_feature_table=None):
    """
    i1: int, the start index of the 1st occluder
    i2: int, the start index of the 2nd occluder
    k: int, the length of occluder.
    """
    #### Prepare occlusion for SVM
    tmp_feature_table=svm_feature_table.loc[RBP_uniprotID]
    
    

    ### decrease corresponding k-mer
    ## 3-mer
    update_kmer_count_2D(RBP_uniprotID, seq, i1, i2, k, 3, tmp_feature_table, mer3_selected) 
    
    ## 4-mer
    update_kmer_count_2D(RBP_uniprotID, seq, i1, i2, k, 4, tmp_feature_table, mer4_selected)
    
    ### decrease corresponding SS-kmer
    ## 11-mer
    update_kmer_count_2D(RBP_uniprotID, ss, i1, i2, k, 11, tmp_feature_table, SSmer11_selected)

    ## 15-mer
    update_kmer_count_2D(RBP_uniprotID, ss, i1, i2, k, 15, tmp_feature_table, SSmer15_selected)

    ### re-run PseAAC
    seq_occ=seq[:i]+seq[(i+k):]
    tmp_paac=get_AAC_features(seq_occ) ## the output is a dictionary
    if not tmp_paac:
        raise ValueError(RBP_uniprotID+"Sequence ("+seq_occ+") gets problem in calculating PseAAC.")
    else:    
        for p in PseAAC_selected:
            tmp_feature_table[p]=tmp_paac[p]

    return np.array(tmp_feature_table)

def get_PseAAC_features(s):
    """
    s: pure sequence of the protein.
    """
    s=s.replace('*','')
    s=s.replace('X','') #solve bug for those cannot be processed by propy
    s=s.replace('U','C') #solve bug for those 'U' that cannot be processed by propy
    try:
        DesObject=PyPro.GetProDes(s)
        paac=DesObject.GetPAAC(lamda=10, weight=0.05)  #the output is a dictionary
    except Exception as e:
        print(type(e))
        print(e, s)
        return None

    return paac


def get_AAC_features(seq, AAs):
    dic = {aa:0 for aa in AAs}
    seq_len=len(seq)
    for aa in seq:
        if aa not in dic.keys():
            dic[aa]=1
        else:
            dic[aa]+=1

    return {k:v*1.0/seq_len for k, v in dic.items()}

def get_AAC_features_bk(s):
    """
    s: pure sequence of the protein.
    """
    s=s.replace('*','')
    s=s.replace('X','') #solve bug for those cannot be processed by propy
    s=s.replace('U','C') #solve bug for those 'U' that cannot be processed by propy
    try:
        DesObject=PyPro.GetProDes(s)
        aac=DesObject.GetPAAC(lamda=0, weight=0.05)  #the output is a dictionary
    except Exception as e:
        print(type(e))
        print(e, s)
        return None

    return aac

def get_occluding_1Dheatmap_domain_stripe_SONARp_SeqOnly(RBP_uniprotID, model_DNN, model_SVM, k=5, no_secondary_structure=False, svm_feature_table=None, scores_DNN=None, scores_SVM=None, true_labels=None, selected_aa3mers=None, selected_aa4mers=None, selected_SS11mers=None, selected_SS15mers=None, selected_AAC=None, sliding_step=1, seq_dir='/home/wjin/data2/proteins/uniport_data/canonical_seq/', BioVec_name_dict=None, AAs=None):
    # file=os.path.join(seq_dir,RBP_uniprotID+'.fasta')
    # file_ss=os.path.join(seq_dir,RBP_uniprotID+'.spd3')
    with open(os.path.join(seq_dir,RBP_uniprotID+'.fasta')) as f:
        prot_seq=''.join(f.read().strip('* \n').split('\n')[1:])
        prot_seq=prot_seq.replace('*','')
        prot_seq=prot_seq.replace('X','')

    if no_secondary_structure==False:
        if os.path.exists(os.path.join(seq_dir,RBP_uniprotID+'.spd3')):
            tmp=pd.read_table(os.path.join(seq_dir,RBP_uniprotID+'.spd3'))
            tmp=tmp[tmp['AA']!='*']
            tmp=tmp[tmp['AA']!='X']
            ss_seq=''.join(list(tmp['SS'])).strip(' ')
        else:
            with open(os.path.join(seq_dir,RBP_uniprotID+'.txt')) as f:
                ss_seq=f.read().strip(' \n').split('\n')[1].strip('*')
        if len(prot_seq)!=len(ss_seq):
            raise ValueError("Amino acids and secondary structure sequence length is not consistent in "+RBP_uniprotID+".")

    ## DNN features
    class_label=1
    if no_secondary_structure:
        RBP=Protein_Sequence_Input5_2_noSS([RBP_uniprotID], [prot_seq], [class_label], BioVec_name_dict, maxlen=1500)
        RBP_aa3mer=RBP.get_aa3mer_mats()[0]
        RBP_seqlens = RBP.get_seqlens()[0]
    else:
        RBP=Protein_Sequence_Input5_2([RBP_uniprotID], [prot_seq], [ss_seq], [class_label], BioVec_name_dict, maxlen=1500)
        RBP_aa3mer=RBP.get_aa3mer_mats()[0]
        RBP_ss_sparse_mat = RBP.get_ss_sparse_mats2()[0]
        RBP_seqlens = RBP.get_seqlens()[0]

    ## SVM features
    tmp_feature_table=svm_feature_table.loc[RBP_uniprotID]

    occluded_DNN_scores=[]
    occluded_SVM_scores=[]
    occluded_ens_scores=[]
    occluded_coord=[]
    if no_secondary_structure:
        original_DNN_score=model_DNN.predict_score([RBP_aa3mer], [RBP_seqlens])[0]
        original_SVM_score=model_SVM.predict_proba([np.array(tmp_feature_table)])[:,1][0]
    else:
        original_DNN_score=model_DNN.predict_score([RBP_aa3mer], [RBP_ss_sparse_mat], [RBP_seqlens])[0]
        original_SVM_score=model_SVM.predict_proba([np.array(tmp_feature_table)])[:,1][0]
    
    original_ens_score=np.array(1-get_model_fdr(scores_DNN, true_labels, original_DNN_score)*get_model_fdr(scores_SVM, true_labels, original_SVM_score))

    ## i=1
    #### occlusion for DNN 
    i=0
    RBP_aa3mer_cp=RBP_aa3mer.copy()
    RBP_aa3mer_cp[i:i+k]=0
    if no_secondary_structure==False:
        RBP_ss_sparse_mat_cp=RBP_ss_sparse_mat.copy()
        RBP_ss_sparse_mat_cp[i:i+k]=[0,0,0]
        y_DNN_pred = model_DNN.predict_score([RBP_aa3mer_cp], [RBP_ss_sparse_mat_cp], [RBP_seqlens])[0]
    else:
        y_DNN_pred = model_DNN.predict_score([RBP_aa3mer_cp], [RBP_seqlens])[0]
    occluded_DNN_scores.append(y_DNN_pred)
    #### Occlusion for SVM
    if no_secondary_structure:
        SVM_features0=get_selected_SVM_features_noSS_1D_occlusion(RBP_uniprotID, i, k, prot_seq, svm_feature_table=svm_feature_table, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_AAC=selected_AAC, AAs=AAs)
    else:
        SVM_features0=get_selected_SVM_features_1D_occlusion(RBP_uniprotID, i, k, prot_seq, ss_seq, svm_feature_table=svm_feature_table, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_SS11mers=selected_SS11mers, selected_SS15mers=selected_SS15mers, selected_AAC=selected_AAC, AAs=AAs)
    y_SVM_pred=model_SVM.predict_proba([np.array(SVM_features0)])[:,1][0]
    occluded_SVM_scores.append(y_SVM_pred)
    occluded_coord.append(str(i)+'-'+str(i+k-1))
    #occluded_ens_scores.append(1-get_model_fdr(scores_DNN, true_labels, y_DNN_pred)*get_model_fdr(scores_SVM, true_labels, y_SVM_pred))

    len_seq=len(prot_seq)
    for i in range(1, len_seq-k+1, sliding_step):
        #### occlusion for DNN 
        RBP_aa3mer_cp=RBP_aa3mer.copy()
        RBP_aa3mer_cp[i:i+k]=0
        if no_secondary_structure:
            y_DNN_pred = model_DNN.predict_score([RBP_aa3mer_cp], [RBP_seqlens])[0]
        else:
            RBP_ss_sparse_mat_cp=RBP_ss_sparse_mat.copy()
            RBP_ss_sparse_mat_cp[i:i+k]=[0,0,0]
            y_DNN_pred = model_DNN.predict_score([RBP_aa3mer_cp], [RBP_ss_sparse_mat_cp], [RBP_seqlens])[0]
            
        occluded_DNN_scores.append(y_DNN_pred)
        #### Occlusion for SVM
        up_count_kmer_2(prot_seq[max(0,i-3):i], SVM_features0, selected_aa3mers)
        down_count_kmer_2(prot_seq[i+k-1:min(i+k+3-1, len_seq)], SVM_features0, selected_aa3mers)
        up_count_kmer_2(prot_seq[max(0,i-4):i], SVM_features0, selected_aa4mers)
        down_count_kmer_2(prot_seq[i+k-1:min(i+k+4-1, len_seq)], SVM_features0, selected_aa4mers)
        if no_secondary_structure==False:
            up_count_kmer_2(ss_seq[max(0,i-11):i], SVM_features0, selected_SS11mers)
            down_count_kmer_2(ss_seq[i+k-1:min(i+k+11-1, len_seq)], SVM_features0, selected_SS11mers)
            up_count_kmer_2(ss_seq[max(0,i-15):i], SVM_features0, selected_SS15mers)
            down_count_kmer_2(ss_seq[i+k-1:min(i+k+15-1, len_seq)], SVM_features0, selected_SS15mers)
            ### re-run PseAAC
        seq_occ=prot_seq[:i]+prot_seq[(i+k):]
        tmp_paac=get_AAC_features(seq_occ, AAs) ## the output is a dictionary
        if not tmp_paac:
            raise ValueError(RBP_uniprotID+"Sequence ("+seq_occ+") gets problem in calculating PseAAC.")
        else:    
            for p in selected_AAC:
                SVM_features0[p]=tmp_paac[p]


        y_SVM_pred=model_SVM.predict_proba([np.array(SVM_features0)])[:,1][0]
        occluded_SVM_scores.append(y_SVM_pred)
        occluded_coord.append(str(i)+'-'+str(i+k-1))
        #occluded_ens_scores.append(1-get_model_fdr(scores_DNN, true_labels, y_DNN_pred)*get_model_fdr(scores_SVM, true_labels, y_SVM_pred))

    return occluded_DNN_scores, occluded_SVM_scores, original_DNN_score, original_SVM_score, occluded_coord


def run_occlusion(RBP_uniprotID, model_DNN, model_SVM, seq_dir, k, out_dir, no_secondary_structure=False, svm_feature_table=None, scores_DNN=None, scores_SVM=None, true_labels=None, selected_aa3mers=None, selected_aa4mers=None, selected_SS11mers=None, selected_SS15mers=None, selected_AAC=None, BioVec_name_dict=None, AAs=None):
    path=out_dir
    if not os.path.exists(path):
        os.mkdir(path)
    if not (os.path.exists(os.path.join(seq_dir, RBP_uniprotID+'.fasta')) and os.path.exists(os.path.join(seq_dir, RBP_uniprotID+'.spd3'))):
        return 
    if not os.path.exists(os.path.join(path, RBP_uniprotID+'_Occlusion_score_matrix_aac.xls')):
        print(RBP_uniprotID)
        occluded_DNN_scores, occluded_SVM_scores, original_DNN_score, original_SVM_score, occluded_coord=get_occluding_1Dheatmap_domain_stripe_SONARp_SeqOnly(RBP_uniprotID, model_DNN, model_SVM, k, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_SS11mers=selected_SS11mers, selected_SS15mers=selected_SS15mers, selected_AAC=selected_AAC, seq_dir=seq_dir, BioVec_name_dict=BioVec_name_dict, AAs=AAs)
        delta_DNN=occluded_DNN_scores - original_DNN_score
        delta_SVM=occluded_SVM_scores - original_SVM_score
        #delta_ens=occluded_ens_scores - original_ens_score

        scores_df=pd.DataFrame.from_dict({'occluded_DNN_scores':occluded_DNN_scores,'occluded_SVM_scores':occluded_SVM_scores, 'original_DNN_score':original_DNN_score, 'original_SVM_score':original_SVM_score, 'delta_DNN':delta_DNN, 'delta_SVM':delta_SVM, 'occluded_coord':occluded_coord})
        scores_df.to_csv(os.path.join(path, RBP_uniprotID+'_Occlusion_score_matrix_aac.xls'), index=True, sep='\t')

def run_ensemble(RBP_uniprotID, out_dir, scores_DNN=None, scores_SVM=None, true_labels=None):
    path = os.path.join(out_dir,RBP_uniprotID+'_Occlusion_score_matrix_aac.xls')
    if os.path.exists(path) and (not os.path.exists(path.replace('_aac.xls','_full_aac.xls'))):
        print(RBP_uniprotID)
        df=pd.read_table(path,index_col=0)
        df['occluded_ens_scores']=df.apply(lambda x: get_ensScore(x['occluded_DNN_scores'],x['occluded_SVM_scores'],scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels), axis=1)
        df['original_ens_score']=df.apply(lambda x: get_ensScore(x['original_DNN_score'],x['original_SVM_score'],scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels), axis=1)
        df['delta_ens']=df.apply(lambda x: x['occluded_ens_scores']-x['original_ens_score'], axis=1)
        df.to_csv(os.path.join(out_dir, RBP_uniprotID+'_Occlusion_score_matrix_full_aac.xls'), index=True, sep='\t')

def plot_occlusion(RBP_uniprotID, out_dir, wind_size, annotation_file=None, annotation_file_separator=','):
    """
    annotation_file: a table with columns of Start, Stop, Type and region_name. The Start and Stop columns contains the 1-based coordinates of the region for annotation. An Example is shown here:
    _______________________________________
    | Start | Stop |  Type  |  Region_name|
    |--------------------------------------
    |   3   |  19  | Domain |      AA     |
    |--------------------------------------
    |  40   |  60  | Domain |      BB     |
    |--------------------------------------
    |  30   |  50  |  IDR   |     None    | 
    |--------------------------------------
    |  70   |  80  |  LC    |     None    |
    |--------------------------------------

    """
    path=os.path.join(out_dir, RBP_uniprotID+'_Occlusion_score_matrix_full_aac.xls')
    delta_SVM=list(pd.read_table(path, index_col=0).delta_SVM)
    delta_DNN=list(pd.read_table(path, index_col=0).delta_DNN)
    delta_ens=list(pd.read_table(path, index_col=0).delta_ens)
    delta_SVM=[0]*int((wind_size-1)/2)+delta_SVM+[0]*int((wind_size)/2)
    delta_DNN=[0]*int((wind_size-1)/2)+delta_DNN+[0]*int((wind_size)/2)
    delta_ens=[0]*int((wind_size-1)/2)+delta_ens+[0]*int((wind_size)/2)

    if annotation_file!=None:
        ann_df=pd.read_csv(annotation_file, sep=annotation_file_separator)
        ann_df.Type=ann_df.Type.apply(lambda x: x.strip(' ').upper())
        ann_df_g=ann_df.groupby('Type')
        types=list(ann_df_g.groups.keys())
        f, ax = plt.subplots(3+len(types), 1, figsize=(20, 3*(3+len(types))))
        cm = plt.cm.get_cmap('Pastel1')
        for i in range(len(types)):
            t=types[i]
            bar=np.zeros(len(delta_ens))
            tmp_df=ann_df_g.get_group(t)
            coords=list(zip(tmp_df.Start, tmp_df.Stop))
            for coord in coords:
                bar[(coord[0]-1):coord[1]]=-i

            bar=np.array([bar]*max(int(len(delta_ens)/80),1))
            im=ax[i].imshow(bar, cmap=cm, vmin=-8, vmax=0)
            ax[i].axes.get_yaxis().set_visible(False)
            ax[i].set_title(RBP_uniprotID+'_'+t,fontsize='x-large',verticalalignment='bottom')

        ax[len(types)].plot(range(len(delta_SVM)),delta_SVM, lw=2)
        ax[len(types)].set_xlim(0,len(delta_SVM))
        ax[len(types)].axhline(y=0,ls='-', color='grey', lw=2)
        ax[len(types)].set_title('Delta Scores in SVM, (Occ - origin)',fontsize='x-large')
        ax[len(types)+1].plot(range(len(delta_DNN)),delta_DNN, lw=2)
        ax[len(types)+1].set_xlim(0,len(delta_DNN))
        ax[len(types)+1].axhline(y=0,ls='-', color='grey', lw=2)
        ax[len(types)+1].set_title('Delta Scores in DNN, (Occ - origin)',fontsize='x-large')
        ax[len(types)+2].plot(range(len(delta_ens)),delta_ens, lw=2)
        ax[len(types)+2].set_xlim(0,len(delta_ens))
        ax[len(types)+2].axhline(y=0,ls='-', color='grey', lw=2)
        ax[len(types)+2].set_title(RBP_uniprotID+'_'+'Delta Scores in Ensemble clf, (Occ - origin)',fontsize='x-large')


    else:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 9)) 
        ax1.plot(range(len(delta_SVM)),delta_SVM, lw=2)
        ax1.set_xlim(0,len(delta_SVM))
        ax1.axhline(y=0,ls='-', color='grey', lw=2)
        ax1.set_title('Delta Scores in SVM, (Occ - origin)',fontsize='x-large')
        ax2.plot(range(len(delta_DNN)),delta_DNN, lw=2)
        ax2.set_xlim(0,len(delta_DNN))
        ax2.axhline(y=0,ls='-', color='grey', lw=2)
        ax2.set_title('Delta Scores in DNN, (Occ - origin)',fontsize='x-large')
        ax3.plot(range(len(delta_ens)),delta_ens, lw=2)
        ax3.set_xlim(0,len(delta_ens))
        ax3.axhline(y=0,ls='-', color='grey', lw=2)
        ax3.set_title(RBP_uniprotID+'_'+'Delta Scores in Ensemble clf, (Occ - origin)',fontsize='x-large')

    plt.tight_layout()
    f.savefig(os.path.join(out_dir,RBP_uniprotID+'_OcclusionMap1D.pdf'),format='pdf')

def main(args):
    no_secondary_structure=args.no_secondary_structure
    k=args.window_size
    maxlen=args.maxlen
    max_seq_len=args.max_seq_len
    BioVec_weights_file=args.BioVec_weights
    seqDNN_modelfile_stru=args.seqDNN_modelfile_stru
    seqDNN_modelfile_weight=args.seqDNN_modelfile_weight
    seqSVM_modelfile=args.seqSVM_modelfile
    seq_files=args.seq_files
    seq_dir=args.seq_dir
    if not seq_files:
        seq_files=os.listdir(seq_dir)
    else:
        seq_files=seq_files.split(',')
    out_dir=args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ## SVM_seq selected feature table
    selected_aa3mers_file=args.selected_aa3mers_file
    selected_aa4mers_file=args.selected_aa4mers_file
    selected_SS11mers_file=args.selected_SS11mers_file
    selected_SS15mers_file=args.selected_SS15mers_file
    selected_AAC_file=args.selected_AAC_file
    combined_selected_feature_file=args.combined_selected_feature_file
    autoEncoder_CNN_file=args.autoEncoder_CNN_file
    autoEncoder_Dense_file=args.autoEncoder_Dense_file
    reference_score_file=args.reference_score_file
    Model_name=args.model_name
    model_dir=args.model_dir

    if model_dir == None:
        if no_secondary_structure:
            if seqDNN_modelfile_stru == None:
                seqDNN_modelfile_stru = pkg_resources.resource_string(__name__, '../pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_noSS_WithNoSSmodel_model_structure.json').decode("utf-8")
            else:
                with open(seqDNN_modelfile_stru) as f:
                    seqDNN_modelfile_stru=f.read()
            if seqDNN_modelfile_weight == None:
                seqDNN_modelfile_weight = pkg_resources.resource_filename(__name__, '../pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_noSS_WithNoSSmodel_model_weights.h5')
            if seqSVM_modelfile == None:
                seqSVM_modelfile = pkg_resources.resource_stream(__name__, '../pre_trained/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_noSS_WithNoSSmodel_ModelFile.pkl')
        else:
            if seqDNN_modelfile_stru == None:
                seqDNN_modelfile_stru = pkg_resources.resource_string(__name__, '../pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_model_structure.json').decode("utf-8")
            else:
                with open(seqDNN_modelfile_stru) as f:
                    seqDNN_modelfile_stru=f.read()
            if seqDNN_modelfile_weight == None:
                seqDNN_modelfile_weight = pkg_resources.resource_filename(__name__, '../pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_model_weights.h5')
            if seqSVM_modelfile == None:
                seqSVM_modelfile = pkg_resources.resource_stream(__name__, '../pre_trained/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_ModelFile.pkl')
    else:
        seqDNN_modelfile_stru=os.path.join(model_dir, Model_name+'_seqDNN_model_structure.json')
        with open(seqDNN_modelfile_stru) as f:
                seqDNN_modelfile_stru=f.read()

        seqDNN_modelfile_weight=os.path.join(model_dir, Model_name+'_seqDNN_model_weights.h5')
        seqSVM_modelfile = os.path.join(model_dir, Model_name+'_seqSVM_ModelFile.pkl')

    if BioVec_weights_file == None:
        BioVec_weights_file = pkg_resources.resource_stream(__name__, '../data/protVec_100d_3grams.csv')


    if model_dir == None:
        if selected_aa3mers_file == None:
            selected_aa3mers_file = pkg_resources.resource_string(__name__, '../data/SVM_mer3_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt').decode("utf-8")
        else:
            with open(selected_aa3mers_file) as f:
                selected_aa3mers_file = f.read() 
        if selected_aa4mers_file == None:
            selected_aa4mers_file = pkg_resources.resource_string(__name__, '../data/SVM_mer4_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt').decode("utf-8")
        else:
            with open(selected_aa4mers_file) as f:
                selected_aa4mers_file = f.read()
        if no_secondary_structure==False:
            if selected_SS11mers_file == None:
                selected_SS11mers_file = pkg_resources.resource_string(__name__, '../data/SVM_SS_11mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt').decode("utf-8")
            else:
                with open(selected_SS11mers_file) as f:
                    selected_SS11mers_file=f.read()
            if selected_SS15mers_file == None:
                selected_SS15mers_file = pkg_resources.resource_string(__name__, '../data/SVM_SS_15mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt').decode("utf-8")
            else:
                with open(selected_SS15mers_file) as f:
                    selected_SS15mers_file=f.read()
        if selected_AAC_file == None:
            selected_AAC_file = pkg_resources.resource_string(__name__, '../data/SVM_AAC_chi2_alpha0.01_selected_features_From_WholeDataSet_AACname.txt').decode("utf-8")
        else:
            with open(selected_AAC_file) as f:
                selected_AAC_file=f.read()
        if combined_selected_feature_file == None:
            combined_selected_feature_file = pkg_resources.resource_stream(__name__, '../data/SVM_SeqFeature_table_comined_all_selected_features_From_WholeDataset_12Apr_AACname.txt')
            combined_selected_feature = pd.read_table(combined_selected_feature_file, index_col=0).drop('RBP_flag',axis=1).columns

        else:
            with open(combined_selected_feature_file) as f:
                    combined_selected_feature=f.read().strip(' \n').split('\n')

    else:
        selected_aa3mers_file = os.path.join(model_dir, Model_name+'_SVM_mer3_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
        with open(selected_aa3mers_file) as f:
            selected_aa3mers_file=f.read()

        selected_aa4mers_file = os.path.join(model_dir, Model_name+'_SVM_mer4_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
        with open(selected_aa4mers_file) as f:
            selected_aa4mers_file=f.read()

        if no_secondary_structure==False:
            selected_SS11mers_file = os.path.join(model_dir, Model_name+'_SVM_SS_11mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
            with open(selected_SS11mers_file) as f:
                selected_SS11mers_file=f.read()
            selected_SS15mers_file = os.path.join(model_dir, Model_name+'_SVM_SS_15mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
            with open(selected_SS15mers_file) as f:
                selected_SS15mers_file=f.read()

        selected_AAC_file = os.path.join(model_dir, Model_name+'_SVM_AAC_chi2_alpha0.01_selected_features_From_WholeDataSet.txt')
        with open(selected_AAC_file) as f:
            selected_AAC_file=f.read()

        if no_secondary_structure==False:
            combined_selected_feature_file = os.path.join(model_dir, Model_name+'_SVM_SeqFeature_all_selected_features_From_WholeDataSet.txt')
        else:
            combined_selected_feature_file = os.path.join(model_dir, Model_name+'_SVM_SeqFeatureNoSS_all_selected_features_From_WholeDataSet.txt')
            
        with open(combined_selected_feature_file) as f:
            combined_selected_feature=f.read().strip(' \n').split('\n')

    if reference_score_file == None:
        if no_secondary_structure:
            reference_score_file = pkg_resources.resource_stream(__name__, '../data/Classification_cv_scores_AllRuns_SONAR+_final_Ensemble_17_3_7_3_combined10RBPlist_noSS_menthaBioPlex_STRING.tsv')
        else:
            reference_score_file = pkg_resources.resource_stream(__name__, '../data/Classification_cv_scores_AllRuns_SONAR+_final_Ensemble_17_3_7_3_combined10RBPlist_menthaBioPlex_STRING.tsv')
    if autoEncoder_CNN_file == None:
        if no_secondary_structure:
            autoEncoder_CNN_file = pkg_resources.resource_filename(__name__, '../pre_trained/CNN_autoencoder_2CNN2MaxPool_64filter5_2poollength_first1500aa_withRBPs_noSS_OOPSXRNAXadded.h5')
        else:
            autoEncoder_CNN_file = pkg_resources.resource_filename(__name__, '../pre_trained/CNN_autoencoder_2CNN2MaxPool_64filter5_2poollength_first1500aa_withRBPs_OOPSXRNAXadded.h5')
    if autoEncoder_Dense_file == None:
        if no_secondary_structure:
            autoEncoder_Dense_file = pkg_resources.resource_filename(__name__, '../pre_trained/DenseI21_5_autoencoder_2CNN2MaxPool_64filter5_2poollength_first1500aa_withRBPs_OOPSXRNAXadded.h5')
        else:
            autoEncoder_Dense_file = pkg_resources.resource_filename(__name__, '../pre_trained/DenseI21_5_autoencoder_2CNN2MaxPool_64filter5_2poollength_first1500aa_withRBPs_noSS_OOPSXRNAXadded.h5')

    ## For naive ensembling (probabilistic)
    AAs=['H','K','D','E','S','T','N','Q','C','G','P','A','V','I','L','M','F','Y','W','R']

    reference_score_df=pd.read_table(reference_score_file, index_col=0)

    scores_DNN=list(reference_score_df['seqDNN_score'])
    scores_SVM=list(reference_score_df['seqSVM_score'])
    true_labels=list(reference_score_df['RBP_flag'])

    aa_code={'H':1, 'K':2, 'D':3, 'E':4,
                     'S':5, 'T':6, 'N':7, 'Q':8, 'C':9,
                     'U':10, 'G':11, 'P':12, 'A':13, 'V':14,
                     'I':15, 'L':16, 'M':17, 'F':18, 'Y':19,
                     'W':20, 'R':21, 'blank':0}
    aa_code_reverse={v:k for k, v in aa_code.items()}
    BioVec_weights=pd.read_table(BioVec_weights_file , sep='\t', header=None, index_col=0)
    BioVec_weights_add_null=np.append(np.zeros((1,100)), BioVec_weights.values, axis=0) #append a [0,0,...,0] array at the top of the matrix, which used for padding 0s.
    BioVec_weights_add_null=BioVec_weights_add_null*10
    BioVec_name_dict={}

    for i in range(1, len(BioVec_weights)+1):
        BioVec_name_dict.update({BioVec_weights.index[i-1]:i})
        
    BioVec_name_reverse_dict={v:k for k, v in BioVec_name_dict.items()}

    RBPs=['.'.join(f.strip(' ').split('.')[:-1]) for f in seq_files if f.endswith('.fasta')]
    if len(RBPs)==0:
        raise FileNotFoundError("Sequence files with suffix '.fasta' are missing.")
    seq_dic={}
    ss_seq_dic={}
    max_seq_len=0
    for prot in RBPs:
        with open(os.path.join(seq_dir,prot+'.fasta')) as f:
            seq=''.join(f.read().strip('* \n').split('\n')[1:])
            if len(seq)>max_seq_len:
                max_seq_len=len(seq)

        if no_secondary_structure==False:
            try:
                if os.path.exists(os.path.join(seq_dir, prot+'.spd3')):
                    tmp=pd.read_table(os.path.join(seq_dir,prot+'.spd3'))
                    tmp=tmp[tmp['AA']!='*']
                    tmp=tmp[tmp['AA']!='X']
                    ss=''.join(list(tmp['SS'])).strip(' ')

                elif os.path.exists(os.path.join(seq_dir, prot+'.txt')):
                    with open(os.path.join(seq_dir, prot+'.txt')) as f:
                        ss=f.read().split('\n')[1]
                else:
                    raise FileNotFoundError("Secondary structure file is missing for {}.".format(prot))
                            
                if len(seq)!=len(ss):
                    raise ValueError("Seq and SS seq length is not consistent in {}.".format(prot))
                
            except ValueError:
                print("The lengths of protein sequnce and Secondary structure sequence are not consistent in {}.".format(prot))
            except FileNotFoundError:
                print("Secondary structure file is missing for {}.".format(prot))
            ss_seq_dic[prot]=ss

        seq_dic[prot]=seq
        
    ## SVM features
    selected_aa3mers=selected_aa3mers_file.split('\n')
    selected_aa4mers=selected_aa4mers_file.split('\n')
    if no_secondary_structure==False:
        selected_SS11mers=selected_SS11mers_file.split('\n')
        selected_SS15mers=selected_SS15mers_file.split('\n')

    selected_AAC=selected_AAC_file.split('\n')
    selected_AAC=[x for x in selected_AAC if x]
      
    # combined_selected_feature = pd.read_table(combined_selected_feature_table, index_col=0).drop('RBP_flag',axis=1).columns

    if no_secondary_structure:
        combined_selected_feature =list(filter(lambda x: len(x)<11, combined_selected_feature))
        svm_feature_table=pd.DataFrame([Get_feature_table_noSS(seq_dic[prot], selected_aa3mers, selected_aa4mers, selected_AAC,combined_selected_feature) for prot in RBPs],index=RBPs, columns=combined_selected_feature)
    else:
        svm_feature_table=pd.DataFrame([Get_feature_table(seq_dic[prot], ss_seq_dic[prot], selected_aa3mers, selected_aa4mers, selected_SS11mers, selected_SS15mers, selected_AAC,combined_selected_feature) for prot in RBPs],index=RBPs, columns=combined_selected_feature)

    
    ## Set up models
    autoEncoder_CNN = load_model(autoEncoder_CNN_file)
    autoEncoder_Dense = load_model(autoEncoder_Dense_file)
    if no_secondary_structure:
        model_DNN=SONARp_DNN_SeqOnly_noSS(BioVec_weights_add_null=BioVec_weights_add_null, CNN_trainable=False, maxlen=maxlen, max_seq_len=max_seq_len, dropout=0.3, class_weight={0:1., 1:11.}, batch_size=128, val_fold=None, sliding_step=int((maxlen-2)/2), n_gpus=1)
    else:
        model_DNN=SONARp_DNN_SeqOnly(BioVec_weights_add_null=BioVec_weights_add_null, CNN_trainable=False, maxlen=maxlen, max_seq_len=max_seq_len, dropout=0.3, class_weight={0:1., 1:11.}, batch_size=128, val_fold=None, sliding_step=int((maxlen-2)/2), n_gpus=1)
    
    model_DNN.load_model2(seqDNN_modelfile_stru, seqDNN_modelfile_weight)

    model_SVM=joblib.load(seqSVM_modelfile)
    model_SVM.kernel='rbf' # To fix, sklearn conflict issue between python3 and python2 model string.


    Path(out_dir).mkdir(parents=True, exist_ok=True)
    shuffle(RBPs)
    for RBP in RBPs:
        if not os.path.exists(os.path.join(out_dir,RBP+'_OcclusionMap1D.pdf')):
            if no_secondary_structure:
                run_occlusion(RBP, model_DNN, model_SVM, seq_dir, k, out_dir, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_AAC=selected_AAC, BioVec_name_dict=BioVec_name_dict, AAs=AAs)
            else:
                run_occlusion(RBP, model_DNN, model_SVM, seq_dir, k, out_dir, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_SS11mers=selected_SS11mers, selected_SS15mers=selected_SS15mers, selected_AAC=selected_AAC, BioVec_name_dict=BioVec_name_dict, AAs=AAs)
            run_ensemble(RBP, out_dir, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels)
            plot_occlusion(RBP, out_dir, k)


    #pool=mp.Pool(processes=16)
    #result=[pool.apply_async(run, args=(RBP, model_SeqOnly, k,)) for RBP in RBPs]
    #results=[p.get() for p in result]


def call_main():
    usage="""\nocclusion_map -s input_dir -f fasta_files """ 
    description="""Use trained HydRa to predict the RNA-binding capacity of given proteins. ps: If at least one of the -p/--PPI_feature_file or -P/--PPI2_feature_file is provided, then PPI_edgelist and PAI_edgelist will be ignored."""
    parser= ArgumentParser(usage=usage, description=description)
    #parser.add_option("-h", "--help", action="help")
    parser.add_argument('--no-secondary-structure', dest='no_secondary_structure', help='Do not use secondary structure information in the model training.', action='store_true')
    parser.add_argument('-w', '--window_size', dest='window_size', type=int ,default=20)
    parser.add_argument('-m', '--maxlen', dest='maxlen', help='', type=int, default=1500)
    parser.add_argument('-M', '--max_seq_len', dest='max_seq_len', help='', type=int, default=1500)
    parser.add_argument('-b', '--BioVec_weights', dest='BioVec_weights', help='', metavar='FILE', default=None)
    parser.add_argument('-f', '--seq_files', dest='seq_files', help='fasta files of the given proteins.', type=str, default=None)
    parser.add_argument('-s', '--seq_dir', dest='seq_dir', help='The directory for processed sequence and secondary structure files. The sequence files should have suffix .fasta and the secondary structure with suffix .spd3 or .txt.', type=str, default=None)
    parser.add_argument('-o', '--out_dir', dest='out_dir', help='directory for output Occlusion Map files. Default: ./OcclusionMap_out.', type=str, default='./OcclusionMap_out')
    parser.add_argument('-D', '--seqDNN_modelfile_stru', dest='seqDNN_modelfile_stru', help='', metavar='FILE', default=None)
    parser.add_argument('-d', '--seqDNN_modelfile_weight', dest='seqDNN_modelfile_weight', help='', metavar='FILE', default=None)
    parser.add_argument('-S', '--seqSVM_modelfile', dest='seqSVM_modelfile', help='', metavar='FILE', default=None)
    parser.add_argument('-r', '--reference_score_file', dest='reference_score_file', help='The file path of the reference score file.', metavar='STRING', default=None)
    parser.add_argument('--autoEncoder_CNN_file', dest='autoEncoder_CNN_file', help='', metavar='FILE', default=None)
    parser.add_argument('--autoEncoder_Dense_file', dest='autoEncoder_Dense_file', help='', metavar='FILE', default=None)
    parser.add_argument('--selected_aa3mers_file', dest='selected_aa3mers_file', help='', metavar='FILE', default=None)
    parser.add_argument('--selected_aa4mers_file', dest='selected_aa4mers_file', help='', metavar='FILE', default=None)
    parser.add_argument('--selected_SS11mers_file', dest='selected_SS11mers_file', help='', metavar='FILE', default=None)
    parser.add_argument('--selected_SS15mers_file', dest='selected_SS15mers_file', help='', metavar='FILE', default=None)
    parser.add_argument('--selected_AAC_file', dest='selected_AAC_file', help='', metavar='FILE', default=None)
    parser.add_argument('--combined_selected_feature_file', dest='combined_selected_feature_file', help='', metavar='FILE', default=None)
    parser.add_argument('-a', '--annotation_file', dest='annotation_file', help='File path of a table with columns of Start, Stop, Type and region_name. The Start and Stop columns contains the 1-based coordinates of the region for annotation.', metavar='FILE', default=None)
    parser.add_argument('--annotation_file_separator', dest='annotation_file_separator', help='File path of a table with columns of Start, Stop, Type and region_name. The Start and Stop columns contains the 1-based coordinates of the region for annotation.', type=str, default=',')
    parser.add_argument('-n', '--model_name', dest='model_name', help='A customized name of this prediction made by the user. This prediction_name will be the prefix of the filenames for the prediction output.', type=str, default='RBP_prediction')
    parser.add_argument('--model_dir', dest='model_dir', help='The filepath of the folder stores all the model files that required to run HydRa.', metavar='FILE', default=None)

    args=parser.parse_args()


    if (args.seq_dir == None) and (args.seq_files == None) :
        parser.error("One of the -s/--seq_dir and -f/--seq_files must be specified.")    
    if args.seq_files!=None and  args.seq_dir != None:
        warnings.warn('The -f/--seq_files has been specified, so that the -s/--seq_dir option will be ignored.')
    if args.model_dir:
        if (args.seqDNN_modelfile_stru!=None) or (args.seqDNN_modelfile_weight!=None) or (args.seqSVM_modelfile!=None) or (args.selected_aa3mers_file!=None) or (args.selected_aa4mers_file!=None) or (args.selected_SS11mers_file!=None) or (args.selected_SS15mers_file!=None) or (args.selected_AAC_file!=None) or (args.combined_selected_feature_file!=None):
            warnings.warn('The --model_dir option has been specified. The model files and selected features list will be extracted from this folder. The other input files for models and selected features will be ignored.')

    print("Occlusion map is started with the trained HydRa model!\n")
    main(args)

if __name__=='__main__':
    call_main()



