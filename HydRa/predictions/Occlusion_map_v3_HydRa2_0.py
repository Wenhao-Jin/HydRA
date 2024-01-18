#!/usr/bin/env python
## v2: simplified the computation of SVM k-mer/SSkmer count updates.
## v3: normalize occlusion scores according to protein length but not orignial hydra score


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
from scipy.stats import norm 
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing as mp
from random import shuffle
import joblib
import copy
from pathlib import Path
from proteinbert import OutputType, OutputSpec, load_pretrained_model_from_dump, conv_and_global_attention_model, InputEncoder
from matplotlib.colors import ListedColormap
from matplotlib import colors
from matplotlib.lines import Line2D

from ..models.Sequence_class import Protein_Sequence_Input5, Protein_Sequence_Input5_2, Protein_Sequence_Input5_2_noSS
from ..models.DNN_seq import SONARp_DNN_SeqOnly, SONARp_DNN_SeqOnly_noSS
from ..preprocessing.get_SVM_seq_features import Get_feature_table, Get_feature_table_noSS

import pkg_resources
import pickle
from pathlib import Path
import warnings
import logging

plt.switch_backend('agg')

FAKE_ZEROS=1e-5
pvalue_cmap = ListedColormap(["white", "lightskyblue", "steelblue"])

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

def get_ensScore(y_DNN_pred, y_SVM_pred, y_ProteinBERT_pred, scores_DNN=None, scores_SVM=None, scores_ProteinBERT=None, true_labels=None):
    """
    Calculate ensemble score of DNN's and SVM's scores using probablity ensembling.
    """
    return 1-get_model_fdr(scores_DNN, true_labels, y_DNN_pred)*get_model_fdr(scores_SVM, true_labels, y_SVM_pred)*get_model_fdr(scores_ProteinBERT, true_labels, y_ProteinBERT_pred)


def merge_peaks(peak_list):
    if len(peak_list)>0:
        out_list=[]
        peak0=peak_list[0]
        start0=int(peak0.split('-')[0])
        end0=int(peak0.split('-')[1])
        for peak in peak_list[1:]:
            start=int(peak.split('-')[0])
            end=int(peak.split('-')[1])
            if start<=end0:
                end0=end
            else:
                out_list.append('{}-{}'.format(start0,end0))
                start0=start
                end0=end
        
        out_list.append('{}-{}'.format(start0,end0))
        return out_list
    else:
        return []

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
                tmp_feature_table[kmer]=0
                warnings.warn("Feature ("+kmer+") counts are less than 0.")
                #raise ValueError("Feature ("+kmer+") counting problems.")


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

    return {k:v*1.0/(seq_len+FAKE_ZEROS) for k, v in dic.items()}

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

def occlude_X_proteinBERT(X, start, end, value=22):
    a=X[0]
    for i in range(start, end):
        a.itemset(i,value)
    
    X[0]=a
    return X

def get_occluding_1Dheatmap_domain_stripe_SONARp_SeqOnly(RBP_uniprotID, model_DNN, model_SVM, k=20, no_secondary_structure=False, svm_feature_table=None, scores_DNN=None, scores_SVM=None, true_labels=None, selected_aa3mers=None, selected_aa4mers=None, selected_SS11mers=None, selected_SS15mers=None, selected_AAC=None, sliding_step=1, seq_dir='/home/wjin/data2/proteins/uniport_data/canonical_seq/', BioVec_name_dict=None, AAs=None):
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
    
    #original_ens_score=np.array(1-get_model_fdr(scores_DNN, true_labels, original_DNN_score)*get_model_fdr(scores_SVM, true_labels, original_SVM_score))

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

def get_occluding_1Dheatmap_domain_stripe_SONARp_SeqOnly_HydRa2(RBP_uniprotID, model_DNN, model_SVM, model_ProteinBERT_generator, ProteinBERT_input_encoder, k=20, no_secondary_structure=False, svm_feature_table=None, scores_DNN=None, scores_SVM=None, scores_ProteinBERT=None, true_labels=None, selected_aa3mers=None, selected_aa4mers=None, selected_SS11mers=None, selected_SS15mers=None, selected_AAC=None, sliding_step=1, seq_dir='/home/wjin/data2/proteins/uniport_data/canonical_seq/', seqfile=None, BioVec_name_dict=None, AAs=None, replace_value=22, start_seq_len_ProteinBERT=512):
    # file=os.path.join(seq_dir,RBP_uniprotID+'.fasta')
    # file_ss=os.path.join(seq_dir,RBP_uniprotID+'.spd3')
    print(RBP_uniprotID)
    if seq_dir!=None:
        with open(os.path.join(seq_dir,RBP_uniprotID+'.fasta')) as f:
            prot_seq=''.join(f.read().strip('* \n').split('\n')[1:])
            prot_seq=prot_seq.replace('*','')
            prot_seq_ProteinBERT=prot_seq
            prot_seq=prot_seq.replace('X','')
    elif seqfile!=None:
        with open(seqfile) as f:
            prot_seq=''.join(f.read().strip('* \n').split('\n')[1:])
            prot_seq=prot_seq.replace('*','')
            prot_seq_ProteinBERT=prot_seq
            prot_seq=prot_seq.replace('X','')
    else:
        raise ValueError('fasta sequence are not found.')

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

    ## ProteinBERT sequence input
    seq_len_proteinBERT = start_seq_len_ProteinBERT*(int((len(prot_seq_ProteinBERT)+2)/start_seq_len_ProteinBERT)+1) # 2 is becuase of ADDED_TOKENS_PER_SEQ = 2 in ProteinBERT models.
    X_proteinBERT = ProteinBERT_input_encoder.encode_X([prot_seq_ProteinBERT], seq_len_proteinBERT)
    model_ProteinBERT = model_ProteinBERT_generator.create_model(seq_len_proteinBERT)

    occluded_DNN_scores=[]
    occluded_SVM_scores=[]
    occluded_ProteinBERT_scores=[]
    #occluded_ens_scores=[]
    occluded_coord=[]
    if no_secondary_structure:
        original_DNN_score=model_DNN.predict_score([RBP_aa3mer], [RBP_seqlens])[0]
        original_SVM_score=model_SVM.predict_proba([np.array(tmp_feature_table)])[:,1][0]
    else:
        original_DNN_score=model_DNN.predict_score([RBP_aa3mer], [RBP_ss_sparse_mat], [RBP_seqlens])[0]
        original_SVM_score=model_SVM.predict_proba([np.array(tmp_feature_table)])[:,1][0]

    original_ProteinBERT_score = model_ProteinBERT.predict(X_proteinBERT, batch_size = 1).flatten()[0]
    
    # original_ens_score=np.array(1-get_model_fdr(scores_DNN, true_labels, original_DNN_score)*get_model_fdr(scores_SVM, true_labels, original_SVM_score)*get_model_fdr(scores_ProteinBERT, true_labels, original_ProteinBERT_score))

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
    #### Occlusion for ProteinBERT
    X_proteinBERT_cp=copy.deepcopy(X_proteinBERT)
    occlude_X_proteinBERT(X_proteinBERT_cp, i, i+k, value=replace_value)
    y_ProteinBERT_pred = model_ProteinBERT.predict(X_proteinBERT_cp, batch_size = 1).flatten()[0]
    occluded_ProteinBERT_scores.append(y_ProteinBERT_pred)

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
        #### Occlusion for ProteinBERT
        X_proteinBERT_cp=copy.deepcopy(X_proteinBERT)
        occlude_X_proteinBERT(X_proteinBERT_cp, i, i+k, value=replace_value)
        y_ProteinBERT_pred = model_ProteinBERT.predict(X_proteinBERT_cp, batch_size = 1).flatten()[0]
        occluded_ProteinBERT_scores.append(y_ProteinBERT_pred)

        occluded_coord.append(str(i)+'-'+str(i+k-1))
        #occluded_ens_scores.append(1-get_model_fdr(scores_DNN, true_labels, y_DNN_pred)*get_model_fdr(scores_SVM, true_labels, y_SVM_pred))

    return occluded_DNN_scores, occluded_SVM_scores, occluded_ProteinBERT_scores, original_DNN_score, original_SVM_score, original_ProteinBERT_score, occluded_coord

def run_occlusion(RBP_uniprotID, model_DNN, model_SVM, seq_dir, k, out_dir, no_secondary_structure=False, svm_feature_table=None, scores_DNN=None, scores_SVM=None, true_labels=None, selected_aa3mers=None, selected_aa4mers=None, selected_SS11mers=None, selected_SS15mers=None, selected_AAC=None, BioVec_name_dict=None, AAs=None):
    path=out_dir
    if not os.path.exists(path):
        os.mkdir(path)

    if not no_secondary_structure:
        if not (os.path.exists(os.path.join(seq_dir, RBP_uniprotID+'.fasta')) and os.path.exists(os.path.join(seq_dir, RBP_uniprotID+'.spd3'))):
            return 
    elif not os.path.exists(os.path.join(seq_dir, RBP_uniprotID+'.fasta')):
        return

    if not os.path.exists(os.path.join(path, RBP_uniprotID+'_Occlusion_score_matrix_aac.xls')):
        print(RBP_uniprotID)
        occluded_DNN_scores, occluded_SVM_scores, original_DNN_score, original_SVM_score, occluded_coord=get_occluding_1Dheatmap_domain_stripe_SONARp_SeqOnly(RBP_uniprotID, model_DNN, model_SVM, k, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, scores_ProteinBERT=scores_ProteinBERT, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_SS11mers=selected_SS11mers, selected_SS15mers=selected_SS15mers, selected_AAC=selected_AAC, seq_dir=seq_dir, BioVec_name_dict=BioVec_name_dict, AAs=AAs)
        delta_DNN=occluded_DNN_scores - original_DNN_score
        delta_SVM=occluded_SVM_scores - original_SVM_score
        #delta_ens=occluded_ens_scores - original_ens_score

        scores_df=pd.DataFrame.from_dict({'occluded_DNN_scores':occluded_DNN_scores,'occluded_SVM_scores':occluded_SVM_scores, 'original_DNN_score':original_DNN_score, 'original_SVM_score':original_SVM_score, 'delta_DNN':delta_DNN, 'delta_SVM':delta_SVM, 'occluded_coord':occluded_coord})
        scores_df.to_csv(os.path.join(path, RBP_uniprotID+'_Occlusion_score_matrix_aac.xls'), index=True, sep='\t')

def run_occlusion_HydRa2(RBP_uniprotID, model_DNN, model_SVM, model_ProteinBERT_generator, ProteinBERT_input_encoder, seq_dir, k, out_dir, seqfile=None, no_secondary_structure=False, svm_feature_table=None, scores_DNN=None, scores_SVM=None, scores_ProteinBERT=None, true_labels=None, selected_aa3mers=None, selected_aa4mers=None, selected_SS11mers=None, selected_SS15mers=None, selected_AAC=None, BioVec_name_dict=None, AAs=None, replace_value=22, start_seq_len_ProteinBERT=512):
    path=out_dir
    if not os.path.exists(path):
        os.mkdir(path)
    if not no_secondary_structure:
        if not (os.path.exists(os.path.join(seq_dir, RBP_uniprotID+'.fasta')) and os.path.exists(os.path.join(seq_dir, RBP_uniprotID+'.spd3'))):
            return 
    elif (seq_dir and (not os.path.exists(os.path.join(seq_dir, RBP_uniprotID+'.fasta')))) or (seqfile and (not os.path.exists(seqfile))):
        return
        
    if not os.path.exists(os.path.join(path, RBP_uniprotID+'_Occlusion_score_matrix_aac.xls')):
        print(RBP_uniprotID)
        occluded_DNN_scores, occluded_SVM_scores, occluded_ProteinBERT_scores, original_DNN_score, original_SVM_score, original_ProteinBERT_score, occluded_coord=get_occluding_1Dheatmap_domain_stripe_SONARp_SeqOnly_HydRa2(RBP_uniprotID, model_DNN, model_SVM, model_ProteinBERT_generator, ProteinBERT_input_encoder, k, seqfile=seqfile, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, scores_ProteinBERT=scores_ProteinBERT, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_SS11mers=selected_SS11mers, selected_SS15mers=selected_SS15mers, selected_AAC=selected_AAC, seq_dir=seq_dir, BioVec_name_dict=BioVec_name_dict, AAs=AAs, replace_value=replace_value, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT)
        delta_DNN=occluded_DNN_scores - original_DNN_score
        delta_SVM=occluded_SVM_scores - original_SVM_score
        delta_ProteinBERT=occluded_ProteinBERT_scores - original_ProteinBERT_score

        #delta_ens=occluded_ens_scores - original_ens_score

        scores_df=pd.DataFrame.from_dict({'occluded_DNN_scores':occluded_DNN_scores,'occluded_SVM_scores':occluded_SVM_scores,'occluded_ProteinBERT_scores':occluded_ProteinBERT_scores, 'original_DNN_score':original_DNN_score, 'original_SVM_score':original_SVM_score, 'original_ProteinBERT_score':original_ProteinBERT_score, 'delta_DNN':delta_DNN, 'delta_SVM':delta_SVM, 'delta_ProteinBERT':delta_ProteinBERT, 'occluded_coord':occluded_coord})
        scores_df.to_csv(os.path.join(path, RBP_uniprotID+'_Occlusion_score_matrix_aac.xls'), index=True, sep='\t')

def get_protein_length(seqfile):
    with open(seqfile) as f:
        seq=f.read().split('\n')[1]
    return len(seq)

def wrap_run_occlusion_HydRa2(RBPs, seq_dir, out_dir, selected_aa3mers_file, selected_aa4mers_file, selected_AAC_file, combined_selected_feature, autoEncoder_CNN_file, autoEncoder_Dense_file, BioVec_weights_add_null, maxlen, max_seq_len, seqDNN_modelfile_stru, seqDNN_modelfile_weight, seqSVM_modelfile, proteinBERT_modelfile, model_dir, model_name, no_secondary_structure, k, BioVec_name_dict, AAs, scores_DNN, scores_SVM, true_labels, selected_SS11mers_file=None, selected_SS15mers_file=None, n_annotations=8943, start_seq_len_ProteinBERT=512, seq_files=None):
    if len(RBPs)==0:
        logging.info("No proteins are input to wrap_run_occlusion_HydRa2 function.")
        return
        #raise FileNotFoundError("Sequence files with suffix '.fasta' are missing.")

    seq_dic={}
    ss_seq_dic={}
    max_seq_len=0
    ava_RBPs=[]
    if (seq_files!=None):
        file_prot_dic={'.'.join(f.strip(' ').split('/')[-1].split('.')[:-1]):f for f in seq_files}
    for prot in RBPs:
        if seq_dir!=None:
            seqfile=os.path.join(seq_dir, prot+'.fasta')
        elif seq_files!=None:
            if prot in file_prot_dic:
                seqfile=file_prot_dic[prot]
            else:
                warnings.warn("Protein {} is skipped because it is not found in seq_files.".format(prot))

        if not os.path.exists(seqfile):
            warnings.warn("{} is not found. It will be ignored in this run.".format(seqfile))
            continue

        if os.path.exists(os.path.join(out_dir, prot+'_Occlusion_score_matrix_aac.xls')):
            logging.info("{}_Occlusion_score_matrix_aac.xls already exists in the output directory. {} will be ignored in the occlusion score generating step.".format(prot, prot))
            continue

        ava_RBPs.append(prot)
        with open(seqfile) as f:
            seq=''.join(f.read().strip('* \n').split('\n')[1:])
            if len(seq)>max_seq_len:
                max_seq_len=len(seq)
        if no_secondary_structure==False:
            try:
                if os.path.exists(seqfile.replace('.fasta','.spd3')):
                    tmp=pd.read_table(seqfile.replace('.fasta','.spd3'))
                    tmp=tmp[tmp['AA']!='*']
                    tmp=tmp[tmp['AA']!='X']
                    ss=''.join(list(tmp['SS'])).strip(' ')

                elif os.path.exists(seqfile.replace('.fasta','.txt')):
                    with open(seqfile.replace('.fasta','.txt')) as f:
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
        svm_feature_table=pd.DataFrame([Get_feature_table_noSS(seq_dic[prot], selected_aa3mers, selected_aa4mers, selected_AAC,combined_selected_feature) for prot in ava_RBPs],index=ava_RBPs, columns=combined_selected_feature)
    else:
        svm_feature_table=pd.DataFrame([Get_feature_table(seq_dic[prot], ss_seq_dic[prot], selected_aa3mers, selected_aa4mers, selected_SS11mers, selected_SS15mers, selected_AAC,combined_selected_feature) for prot in ava_RBPs],index=ava_RBPs, columns=combined_selected_feature)

    
    ## Set up models
    ## seqDNN
    autoEncoder_CNN = load_model(autoEncoder_CNN_file)
    autoEncoder_Dense = load_model(autoEncoder_Dense_file)
    if no_secondary_structure:
        model_DNN=SONARp_DNN_SeqOnly_noSS(BioVec_weights_add_null=BioVec_weights_add_null, CNN_trainable=False, maxlen=maxlen, max_seq_len=max_seq_len, dropout=0.3, class_weight={0:1., 1:11.}, batch_size=128, val_fold=None, sliding_step=int((maxlen-2)/2), n_gpus=1)
    else:
        model_DNN=SONARp_DNN_SeqOnly(BioVec_weights_add_null=BioVec_weights_add_null, CNN_trainable=False, maxlen=maxlen, max_seq_len=max_seq_len, dropout=0.3, class_weight={0:1., 1:11.}, batch_size=128, val_fold=None, sliding_step=int((maxlen-2)/2), n_gpus=1)
    
    model_DNN.load_model2(seqDNN_modelfile_stru, seqDNN_modelfile_weight)

    ## seqSVM
    # print(seqSVM_modelfile)
    try:
        model_SVM=joblib.load(seqSVM_modelfile)
    except:
        logging.info('seqSVM model loading failed, trying with default model file.')
        if no_secondary_structure:
            seqSVM_modelfile = pkg_resources.resource_stream(__name__, '../pre_trained/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_noSS_WithNoSSmodel_ModelFile.pickle.pkl')
        else:
            seqSVM_modelfile = pkg_resources.resource_stream(__name__, '../pre_trained/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_ModelFile.pickle.pkl')
        # print(seqSVM_modelfile)
        # model_SVM = pickle.load(seqSVM_modelfile)
        model_SVM=joblib.load(seqSVM_modelfile)
        warnings.warn('Failed in loading seqSVM model. The pre-trained seqSVM model in our package is used.\n')
        # model_SVM=pickle.load(open(seqSVM_modelfile, 'rb'))
        # model_SVM = pd.read_pickle(seqSVM_modelfile)
        # model_SVM = pickle.load(seqSVM_modelfile)
        # print(model_SVM)

    model_SVM.kernel='rbf' # To fix, sklearn conflict issue between python3 and python2 model string.

    ## ProteinBERT    

    #model_ProteinBERT_generator = load_pretrained_model_from_dump(proteinBERT_modelfile, conv_and_global_attention_model.create_model)
    with open(proteinBERT_modelfile,'rb') as f:
        model_ProteinBERT_generator=pickle.load(f)
    ProteinBERT_input_encoder = InputEncoder(n_annotations)
    ## make reference occlusion scores:
    # for RBP in ava_RBPs:
    #     if not os.path.exists(os.path.join(out_dir,RBP+'_OcclusionMap1D.pdf')):
    #         if no_secondary_structure:
    #             run_occlusion(RBP, model_DNN, model_SVM, seq_dir, k, out_dir, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_AAC=selected_AAC, BioVec_name_dict=BioVec_name_dict, AAs=AAs)
    #         else:
    #             run_occlusion(RBP, model_DNN, model_SVM, seq_dir, k, out_dir, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_SS11mers=selected_SS11mers, selected_SS15mers=selected_SS15mers, selected_AAC=selected_AAC, BioVec_name_dict=BioVec_name_dict, AAs=AAs)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    shuffle(ava_RBPs)
    for RBP in ava_RBPs:
        if not os.path.exists(os.path.join(out_dir,RBP+'_OcclusionMap1D.pdf')):
            if no_secondary_structure:
                if seq_dir!=None:
                    run_occlusion_HydRa2(RBP, model_DNN, model_SVM, model_ProteinBERT_generator, ProteinBERT_input_encoder, seq_dir, k, out_dir, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_AAC=selected_AAC, BioVec_name_dict=BioVec_name_dict, AAs=AAs, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT)
                elif seq_files!=None:
                    if RBP in file_prot_dic:
                        seqfile=file_prot_dic[RBP]
                        run_occlusion_HydRa2(RBP, model_DNN, model_SVM, model_ProteinBERT_generator, ProteinBERT_input_encoder, None, k, out_dir, seqfile=seqfile, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_AAC=selected_AAC, BioVec_name_dict=BioVec_name_dict, AAs=AAs, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT)
            else:
                if seq_dir!=None:
                    run_occlusion_HydRa2(RBP, model_DNN, model_SVM, model_ProteinBERT_generator, ProteinBERT_input_encoder, seq_dir, k, out_dir, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_SS11mers=selected_SS11mers, selected_SS15mers=selected_SS15mers, selected_AAC=selected_AAC, BioVec_name_dict=BioVec_name_dict, AAs=AAs, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT)
                elif seqfiles!=None:
                    if RBP in file_prot_dic:
                        seqfile=file_prot_dic[RBP]
                        run_occlusion_HydRa2(RBP, model_DNN, model_SVM, model_ProteinBERT_generator, ProteinBERT_input_encoder, None, k, out_dir, seqfile=seqfile, no_secondary_structure=no_secondary_structure, svm_feature_table=svm_feature_table, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels, selected_aa3mers=selected_aa3mers, selected_aa4mers=selected_aa4mers, selected_SS11mers=selected_SS11mers, selected_SS15mers=selected_SS15mers, selected_AAC=selected_AAC, BioVec_name_dict=BioVec_name_dict, AAs=AAs, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT)


def get_reference_scores(HydRa_df, occ_dir, SVM_col='seqSVM_score', DNN_col='seqDNN_score'):
    cutoff=1.0
    scores_dic={}
    scores_dic[SVM_col]={}
    scores_dic[DNN_col]={}
    while cutoff>=0.05:
        prot_set_SVM=list(HydRa_df[(HydRa_df[SVM_col]<=cutoff) & (HydRa_df[SVM_col]>cutoff-0.05)].index)
        prot_set_SVM=[x for x in prot_set_SVM if os.path.exists(os.path.join(occ_dir, x+'_Occlusion_score_matrix_full_aac.xls'))]
        prot_set_DNN=list(HydRa_df[(HydRa_df[DNN_col]<=cutoff) & (HydRa_df[DNN_col]>cutoff-0.05)].index)
        prot_set_DNN=[x for x in prot_set_DNN if os.path.exists(os.path.join(occ_dir, x+'_Occlusion_score_matrix_full_aac.xls'))]
        scores_dic[SVM_col][str(int(round(cutoff-0.05,2)*100))+'-'+str(int(cutoff*100))]=[]
        scores_dic[DNN_col][str(int(round(cutoff-0.05,2)*100))+'-'+str(int(cutoff*100))]=[]
        for prot in prot_set_DNN:
            scores_dic[DNN_col][str(int(round(cutoff-0.05,2)*100))+'-'+str(int(cutoff*100))]+=list(pd.read_table(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac.xls'))['delta_DNN'])
        for prot in prot_set_SVM:
            scores_dic[SVM_col][str(int(round(cutoff-0.05,2)*100))+'-'+str(int(cutoff*100))]+=list(pd.read_table(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac.xls'))['delta_SVM'])
        cutoff=round(cutoff-0.05,2)
        
    return scores_dic

def get_reference_avgZscore(HydRa_df, occ_dir, col='avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT'):
    cutoff=1.0
    scores_list=[]
    prot_set=set(HydRa_df.index)
    prot_set=[x for x in prot_set if os.path.exists(os.path.join(occ_dir, x+'_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib.xls'))]
    for prot in prot_set:
        scores_list+=list(pd.read_table(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib.xls'))[col])
    return scores_list
    

def add_avg_zscore_pvalue(df, ref_mean, ref_std, avg_col='avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT'):
    df[avg_col+'_pvalue']=df[avg_col].apply(lambda x: norm.cdf(x, loc=ref_mean, scale=ref_std))
    df['deltaDNN_zscore_pvalue']=df['zscore_deltaDNN'].apply(lambda x: norm.cdf(x))
    df['deltaSVM_zscore_pvalue']=df['zscore_deltaSVM'].apply(lambda x: norm.cdf(x))
    df['deltaProteinBERT_zscore_pvalue']=df['zscore_deltaProteinBERT'].apply(lambda x: norm.cdf(x))
    return df

def generate_peak_df(occ_df, threshold, pvalue_col='avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue'):
    peaks_coords=occ_df[occ_df[pvalue_col]<=threshold][[pvalue_col,'occluded_coord']] ## The original coordinates are 0-based.
    peaks_coords['start']=peaks_coords['occluded_coord'].apply(lambda x: int(x.split('-')[0]))
    peaks_coords['end']=peaks_coords['occluded_coord'].apply(lambda x: int(x.split('-')[1]))
    peaks_coords_df=peaks_coords[['start','end',pvalue_col]]
    peaks_merge=[]
    peaks_coords_list=peaks_coords_df.to_numpy()
    if len(peaks_coords_list)>0:
        start0, end0, p_value0=peaks_coords_list[0]
        if len(peaks_coords_list)>1:
            for start, end, p_value in peaks_coords_list[1:]:
                if start<=end0:
                    start0=start0
                    end0=end
                    p_value0=min(p_value0, p_value)
                else:
                    peaks_merge.append([start0, end0, p_value0])
                    start0=start
                    end0=end
            
            peaks_merge.append([start0, end0, p_value0])
        else:
            peaks_merge.append([start0, end0, p_value0])
            
    peaks_merge_df=pd.DataFrame(peaks_merge, columns=['start','end','min_of_{}s'.format(pvalue_col)])
    return peaks_coords_df, peaks_merge_df
    
def get_z_scores_for_occluder(delta_score, ref_mean, ref_std):
    return (delta_score-ref_mean)*1.0/ref_std

def get_stats_dic(scores_dic):
    stats_dic={}
    for col in scores_dic.keys():
        stats_dic[col]={}
        for k, v in scores_dic[col].items():
            stats_dic[col][k]={}
            stats_dic[col][k]['mean']=np.nanmean(v)
            stats_dic[col][k]['std']=np.nanstd(v)
    
    return stats_dic
def add_z_scores(df, stats_dic, prot_len, SVM_col='seqSVM_score', DNN_col='seqDNN_score', ProteinBERT_col='ProteinBERT_score', avg_col='avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT'):
    if prot_len<=100:
        upper_bound=100
        lower_bound=0
    else:
        for k in stats_dic[DNN_col].keys():
            if k!='all':
                kk=k.split('-')
                u=int(kk[1]) if kk[1] != 'inf' else np.inf
                l=int(kk[0])
                if prot_len>l and prot_len<=u:
                    upper_bound=u
                    lower_bound=l
                    break

    ref_stat_DNN=stats_dic[DNN_col]['{}-{}'.format(lower_bound,upper_bound)]
    mean_DNN=ref_stat_DNN['mean']
    std_DNN=ref_stat_DNN['std']
    if np.isnan(mean_DNN) or np.isnan(std_DNN):
        mean_DNN=stats_dic[DNN_col]['all']['mean']
        std_DNN=stats_dic[DNN_col]['all']['std']

    ref_stat_SVM=stats_dic[SVM_col]['{}-{}'.format(lower_bound,upper_bound)]
    mean_SVM=ref_stat_SVM['mean']
    std_SVM=ref_stat_SVM['std']
    if np.isnan(mean_SVM) or np.isnan(std_SVM):
        mean_SVM=stats_dic[SVM_col]['all']['mean']
        std_SVM=stats_dic[SVM_col]['all']['std']

    ref_stat_ProteinBERT=stats_dic[ProteinBERT_col]['{}-{}'.format(lower_bound,upper_bound)]
    mean_ProteinBERT=ref_stat_ProteinBERT['mean']
    std_ProteinBERT=ref_stat_ProteinBERT['std']
    if np.isnan(mean_ProteinBERT) or np.isnan(std_ProteinBERT):
        mean_ProteinBERT=stats_dic[ProteinBERT_col]['all']['mean']
        std_ProteinBERT=stats_dic[ProteinBERT_col]['all']['std']
    
    df['zscore_deltaSVM']=df.apply(lambda x: get_z_scores_for_occluder(x['delta_SVM'], mean_SVM, std_SVM), axis=1)
    df['zscore_deltaDNN']=df.apply(lambda x: get_z_scores_for_occluder(x['delta_DNN'], mean_DNN, std_DNN), axis=1)
    df['zscore_deltaProteinBERT']=df.apply(lambda x: get_z_scores_for_occluder(x['delta_ProteinBERT'], mean_ProteinBERT, std_ProteinBERT), axis=1)
    df[avg_col]=(df['zscore_deltaSVM']+df['zscore_deltaDNN']+df['zscore_deltaProteinBERT'])/3.0
    
    return df

def add_z_scores_fdrEnsemble(df, stats_dic, SVM_col='seqSVM_score', DNN_col='seqDNN_score', ProteinBERT_col='ProteinBERT_score', seq_score_col='optimistic_DNNSVMnoSSProteinBERT', delta_ens_col='delta_fdr_ens', ens_delta_zscore_col='zscore_deltaFdrEns'):
    orig_DNN=list(df['original_DNN_score'])[0]
    orig_SVM=list(df['original_SVM_score'])[0]
    orig_ProteinBERT=list(df['original_ProteinBERT_score'])[0]
    orig_ens=list(df['original_fdr_ens_score'])[0]

    if (orig_DNN*100)%5==0:
        upper_bound=int(orig_DNN*100)
        lower_bound=upper_bound-5
    else:
        lower_bound=int(orig_DNN*100.0/5)*5
        upper_bound=lower_bound+5
    ref_stat_DNN=stats_dic[DNN_col]['{}-{}'.format(lower_bound,upper_bound)]
    mean_DNN=ref_stat_DNN['mean']
    std_DNN=ref_stat_DNN['std']
    if np.isnan(mean_DNN) or np.isnan(std_DNN):
        mean_DNN=stats_dic[DNN_col]['all']['mean']
        std_DNN=stats_dic[DNN_col]['all']['std']
    
    if (orig_SVM*100)%5==0:
        upper_bound=int(orig_SVM*100)
        lower_bound=upper_bound-5
    else:
        lower_bound=int(orig_SVM*100.0/5)*5
        upper_bound=lower_bound+5
    ref_stat_SVM=stats_dic[SVM_col]['{}-{}'.format(lower_bound,upper_bound)]
    mean_SVM=ref_stat_SVM['mean']
    std_SVM=ref_stat_SVM['std']
    if np.isnan(mean_SVM) or np.isnan(std_SVM):
        mean_SVM=stats_dic[SVM_col]['all']['mean']
        std_SVM=stats_dic[SVM_col]['all']['std']

    if (orig_ProteinBERT*100)%5==0:
        upper_bound=int(orig_ProteinBERT*100)
        lower_bound=upper_bound-5
    else:
        lower_bound=int(orig_ProteinBERT*100.0/5)*5
        upper_bound=lower_bound+5
    ref_stat_ProteinBERT=stats_dic[ProteinBERT_col]['{}-{}'.format(lower_bound,upper_bound)]
    mean_ProteinBERT=ref_stat_ProteinBERT['mean']
    std_ProteinBERT=ref_stat_ProteinBERT['std']
    if np.isnan(mean_ProteinBERT) or np.isnan(std_ProteinBERT):
        mean_ProteinBERT=stats_dic[ProteinBERT_col]['all']['mean']
        std_ProteinBERT=stats_dic[ProteinBERT_col]['all']['std']

    if (orig_ens*100)%5==0:
        upper_bound=int(orig_ens*100)
        lower_bound=upper_bound-5
    else:
        lower_bound=int(orig_ens*100.0/5)*5
        upper_bound=lower_bound+5
    ref_stat_ens=stats_dic[seq_score_col]['{}-{}'.format(lower_bound,upper_bound)]
    mean_ens=ref_stat_ens['mean']
    std_ens=ref_stat_ens['std']
    if np.isnan(mean_ens) or np.isnan(std_ens):
        mean_ens=stats_dic[seq_score_col]['all']['mean']
        std_ens=stats_dic[seq_score_col]['all']['std']
    
    
    df['zscore_deltaSVM']=df.apply(lambda x: get_z_scores_for_occluder(x['delta_SVM'], mean_SVM, std_SVM), axis=1)
    df['zscore_deltaDNN']=df.apply(lambda x: get_z_scores_for_occluder(x['delta_DNN'], mean_DNN, std_DNN), axis=1)
    df['zscore_deltaProteinBERT']=df.apply(lambda x: get_z_scores_for_occluder(x['delta_ProteinBERT'], mean_ProteinBERT, std_ProteinBERT), axis=1)
    df[ens_delta_zscore_col]=df.apply(lambda x: get_z_scores_for_occluder(x[delta_ens_col], mean_ens, std_ens), axis=1)
    df['deltaDNN_zscore_pvalue']=df['zscore_deltaDNN'].apply(lambda x: norm.cdf(x))
    df['deltaSVM_zscore_pvalue']=df['zscore_deltaSVM'].apply(lambda x: norm.cdf(x))
    df['deltaProteinBERT_zscore_pvalue']=df['zscore_deltaProteinBERT'].apply(lambda x: norm.cdf(x))
    df[ens_delta_zscore_col+'_pvalue']=df[ens_delta_zscore_col].apply(lambda x: norm.cdf(x, loc=mean_ens, scale=std_ens))
    return df
    
def prepare_z_score_for_fdrEnsemble(occ_dir, stats_dic, seq_score_col, delta_ens_col='delta_fdr_ens', ens_delta_zscore_col='zscore_deltaFdrEns',peak_pval_threshold=0.05):
    prot_set=list(filter(lambda x: x.endswith('_Occlusion_score_matrix_full_aac_fdrEnsemble.xls'), os.listdir(occ_dir)))
    prot_set=list(set(map(lambda x: x.replace('_Occlusion_score_matrix_full_aac_fdrEnsemble.xls',''), prot_set)))
    for prot in prot_set:
        occ_df=pd.read_table(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac_fdrEnsemble.xls'),index_col=0)
        occ_df=add_z_scores_fdrEnsemble(occ_df, stats_dic, seq_score_col, delta_ens_col=delta_ens_col, ens_delta_zscore_col=ens_delta_zscore_col)
        occ_df.to_csv(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac_fdrEnsemble_zscore_pvalue.xls'), sep='\t',index=True)
        peaks_coords_df, peaks_merge_df = generate_peak_df(occ_df, peak_pval_threshold, pvalue_col=ens_delta_zscore_col+'_pvalue')
        peaks_coords_df.to_csv(os.path.join(occ_dir, prot+'_Occlusion_Map_fdrEnsemble_sig_peaks_all_pvalues{}.csv'.format(peak_pval_threshold)),index=False)
        peaks_merge_df.to_csv(os.path.join(occ_dir, prot+'_Occlusion_Map_fdrEnsemble_sig_peaks_all_merged_pvalues{}.csv'.format(peak_pval_threshold)),index=False)

def get_zscore(seqSVM_seqDNN_score_stats_dic, occ_ref_dir, occ_dir, SVM_col, DNN_col, ProteinBERT_col, prot_len_dic):
    # prot_set=list(filter(lambda x: x.endswith('_Occlusion_score_matrix_full_aac.xls'), os.listdir(occ_dir)))
    # prot_set=list(set(map(lambda x: x.replace('_Occlusion_score_matrix_full_aac.xls',''), prot_set)))
    prot_set=list(filter(lambda x: x.endswith('_Occlusion_score_matrix_aac.xls'), os.listdir(occ_dir))) # and (x.split('_')[0] in HydRa_score_whole_df.index)
    prot_set=list(set(map(lambda x: x.replace('_Occlusion_score_matrix_aac.xls',''), prot_set)))

    for prot in prot_set:
        # occ_df=pd.read_table(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac.xls'),index_col=0)
        occ_df=pd.read_table(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_aac.xls'),index_col=0)
        outfile=os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib.xls')
        if (not os.path.exists(outfile)) and (prot in prot_len_dic):
            occ_df=add_z_scores(occ_df, seqSVM_seqDNN_score_stats_dic, prot_len_dic[prot], SVM_col, DNN_col, ProteinBERT_col, 'avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT')
            occ_df.to_csv(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib.xls'), sep='\t',index=True)

def generate_avg_zscore_stats_dic(HydRa_score_whole_df, occ_ref_dir, col='avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT'):
    avg_zscore_list=get_reference_avgZscore(HydRa_score_whole_df, occ_ref_dir, col)
    mean_ref=np.mean(avg_zscore_list)
    std_ref=np.std(avg_zscore_list)
    return {'mean':mean_ref, 'std':std_ref}

# def get_avg_zscore_stats(HydRa_score_whole_df, seqSVM_seqDNN_score_stats_dic, occ_ref_dir, occ_dir, SVM_col, DNN_col, ProteinBERT_col, prot_len_dic):
#     # prot_set=list(filter(lambda x: x.endswith('_Occlusion_score_matrix_full_aac.xls'), os.listdir(occ_dir)))
#     # prot_set=list(set(map(lambda x: x.replace('_Occlusion_score_matrix_full_aac.xls',''), prot_set)))
#     prot_set=list(filter(lambda x: x.endswith('_Occlusion_score_matrix_aac.xls'), os.listdir(occ_dir))) # and (x.split('_')[0] in HydRa_score_whole_df.index)
#     prot_set=list(set(map(lambda x: x.replace('_Occlusion_score_matrix_aac.xls',''), prot_set)))

#     for prot in prot_set:
#         # occ_df=pd.read_table(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac.xls'),index_col=0)
#         occ_df=pd.read_table(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_aac.xls'),index_col=0)
#         outfile=os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib.xls')
#         if (not os.path.exists(outfile)) and (prot in prot_len_dic):
#             occ_df=add_z_scores(occ_df, seqSVM_seqDNN_score_stats_dic, prot_len_dic[prot], SVM_col, DNN_col, ProteinBERT_col, 'avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT')
#             occ_df.to_csv(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib.xls'), sep='\t',index=True)
        
#     ## collect reference score for avg z-scores
#     avg_zscore_stats_dic=generate_avg_zscore_stats_dic(HydRa_score_whole_df, occ_ref_dir, 'avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT')
#     return avg_zscore_stats_dic

def get_avg_zscore_stats(HydRa_score_whole_df, seqSVM_seqDNN_score_stats_dic, occ_ref_dir, occ_dir, SVM_col, DNN_col, ProteinBERT_col, prot_len_dic):
    get_zscore(seqSVM_seqDNN_score_stats_dic, occ_ref_dir, occ_dir, SVM_col, DNN_col, ProteinBERT_col, prot_len_dic)
    ## collect reference score for avg z-scores
    avg_zscore_stats_dic=generate_avg_zscore_stats_dic(HydRa_score_whole_df, occ_ref_dir, 'avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT')
    return avg_zscore_stats_dic

def prepare_z_score(avg_zscore_stats_dic, occ_dir, avg_col='avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT', peak_pval_threshold=0.05):            
    ## collect reference score for avg z-scores
    #avg_zscore_stats_dic=get_avg_zscore_stats(HydRa_score_whole_df, seqSVM_seqDNN_score_stats_dic, occ_ref_dir, occ_dir, SVM_col, DNN_col)
    ## generate p-values for each delta_SVM and delta_DNN, and the avg z-scores.
    prot_set=list(filter(lambda x: x.endswith('_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib.xls'), os.listdir(occ_dir)))
    prot_set=list(set(map(lambda x: x.replace('_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib.xls',''), prot_set)))
    mean_ref=avg_zscore_stats_dic['mean']
    std_ref=avg_zscore_stats_dic['std']
    for prot in prot_set:
        f=os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib_pvalues.xls')
        if not os.path.exists(f):
            occ_df=pd.read_table(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib.xls'),index_col=0)
            occ_df=add_avg_zscore_pvalue(occ_df, mean_ref, std_ref, avg_col)
            occ_df.to_csv(os.path.join(occ_dir, prot+'_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib_pvalues.xls'), sep='\t',index=True)
        else:
            occ_df=pd.read_table(f,index_col=0)
        peaks_coords_df, peaks_merge_df = generate_peak_df(occ_df, peak_pval_threshold)
        peaks_coords_df.to_csv(os.path.join(occ_dir, prot+'_Occlusion_Map_sig_peaks_all_pvalues{}.csv'.format(peak_pval_threshold)),index=False)
        peaks_merge_df.to_csv(os.path.join(occ_dir, prot+'_Occlusion_Map_sig_peaks_all_merged_pvalues{}.csv'.format(peak_pval_threshold)),index=False)
        
    return avg_zscore_stats_dic

def wrap_run_ensemble(RBPs, out_dir, scores_DNN=None, scores_SVM=None, scores_ProteinBERT=None, true_labels=None):
    for RBP in RBPs:
        run_ensemble(RBP, out_dir, scores_DNN=scores_DNN, scores_SVM=scores_SVM, scores_ProteinBERT=scores_ProteinBERT, true_labels=true_labels)

def run_ensemble(RBP_uniprotID, out_dir, scores_DNN=None, scores_SVM=None, scores_ProteinBERT=None, true_labels=None):
    path = os.path.join(out_dir,RBP_uniprotID+'_Occlusion_score_matrix_aac.xls')
    if os.path.exists(path) and (not os.path.exists(path.replace('_aac.xls','_full_aac_fdrEnsemble.xls'))):
        print(RBP_uniprotID)
        df=pd.read_table(path,index_col=0)
        df['occluded_fdr_ens_scores']=df.apply(lambda x: get_ensScore(x['occluded_DNN_scores'],x['occluded_SVM_scores'],x['occluded_ProteinBERT_scores'],scores_DNN=scores_DNN, scores_SVM=scores_SVM, scores_ProteinBERT=scores_ProteinBERT, true_labels=true_labels), axis=1)
        df['original_fdr_ens_score']=df.apply(lambda x: get_ensScore(x['original_DNN_score'],x['original_SVM_score'],x['original_ProteinBERT_score'],scores_DNN=scores_DNN, scores_SVM=scores_SVM, scores_ProteinBERT=scores_ProteinBERT, true_labels=true_labels), axis=1)
        df['delta_fdr_ens']=df.apply(lambda x: x['occluded_fdr_ens_scores']-x['original_fdr_ens_score'], axis=1)
        df.to_csv(os.path.join(out_dir, RBP_uniprotID+'_Occlusion_score_matrix_full_aac_fdrEnsemble.xls'), index=True, sep='\t')

def plot_occlusion(RBP_uniprotID, out_dir, wind_size, annotation_file=None, annotation_file_separator=',', ens_delta_zscore_col='avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT', ens_delta_zscore_track_name='Avg. Standardized Delta Scores',run_fdr_ensemble=False, p_threshold1=0.05, p_threshold2=0.001, draw_ensemble_only=False):
    """
    annotation_file: a table with columns of Start, Stop, Type and region_name. The Start and Stop columns contains the 1-based coordinates of the region for annotation. An Example is shown here:
    ___________________________________________________________
    | Protein | Start | Stop |  Type  |  Region_name| Color  |
    |---------------------------------------------------------
    | P24785  |   3   |  19  | Domain |      AA     |        |
    |---------------------------------------------------------
    | P24785  |  40   |  60  | Domain |      BB     |        |
    |---------------------------------------------------------
    | P24785  |  30   |  50  |  IDR   |             |        |
    |---------------------------------------------------------
    | Q9W0S7  |  70   |  80  |  LC    |             |  None  |
    |---------------------------------------------------------

    """
    print('-------------------------------')
    print(RBP_uniprotID)
    if not run_fdr_ensemble:
        path=os.path.join(out_dir,RBP_uniprotID+'_Occlusion_score_matrix_full_aac_addZscoresProtLenWiseFib_pvalues.xls')
    else:
        path=os.path.join(out_dir,RBP_uniprotID+'_Occlusion_score_matrix_full_aac_fdrEnsemble_zscore_pvalue.xls')
    occ_df=pd.read_table(path, index_col=0)
    orig_SVM=list(occ_df.original_SVM_score)[0]
    orig_DNN=list(occ_df.original_DNN_score)[0]
    orig_ProteinBERT=list(occ_df.original_ProteinBERT_score)[0]

    delta_SVM=list(occ_df.zscore_deltaSVM)
    delta_DNN=list(occ_df.zscore_deltaDNN)
    delta_ProteinBERT=list(occ_df.zscore_deltaProteinBERT)
    delta_ens=list(occ_df[ens_delta_zscore_col])
    delta_SVM_sig_coords=np.where(np.array(occ_df.deltaSVM_zscore_pvalue)<0.01)[0]+int((wind_size-1)/2)
    delta_DNN_sig_coords=np.where(np.array(occ_df.deltaDNN_zscore_pvalue)<0.01)[0]+int((wind_size-1)/2)
    delta_ProteinBERT_sig_coords=np.where(np.array(occ_df.deltaProteinBERT_zscore_pvalue)<0.01)[0]+int((wind_size-1)/2)
    delta_ens_sig_coords=np.where(np.array(occ_df[ens_delta_zscore_col+'_pvalue'])<0.01)[0]+int((wind_size-1)/2)
    delta_SVM=[0]*int((wind_size-1)/2)+delta_SVM+[0]*int((wind_size)/2)
    delta_DNN=[0]*int((wind_size-1)/2)+delta_DNN+[0]*int((wind_size)/2)
    delta_ProteinBERT=[0]*int((wind_size-1)/2)+delta_ProteinBERT+[0]*int((wind_size)/2)
    delta_ens=[0]*int((wind_size-1)/2)+delta_ens+[0]*int((wind_size)/2)

    if annotation_file!=None:
        ann_df=pd.read_csv(annotation_file, sep=annotation_file_separator)
        ann_df=ann_df[ann_df['Protein']==RBP_uniprotID]
        ann_df.Type=ann_df.Type.apply(lambda x: x.strip(' ').upper())
        ann_df_g=ann_df.groupby('Type')
        types=list(ann_df_g.groups.keys())
        if draw_ensemble_only == True:
            f, ax = plt.subplots(2+len(types), 1, figsize=(20, 3*(2+len(types))))
        else:
            f, ax = plt.subplots(5+len(types), 1, figsize=(20, 3*(5+len(types))))
        cm = plt.cm.get_cmap('Pastel1')
        for i in range(len(types)):
            t=types[i]
            tmp_df=ann_df_g.get_group(t)
            if sum(tmp_df.Color.apply(pd.isnull))>0: ## if at least of the color value of this type is empty, we will use the Pastel1 colormap to determine the colors.
                color_flag=False
            else:
                color_flag=True # using customized colors
            
            if sum(tmp_df.Color.apply(pd.isnull))>0: ## if at least of the color value of this type is empty, we will use the Pastel1 colormap to determine the colors.
                color_flag=False
            else:
                color_flag=True # using customized colors
            coords=list(zip(tmp_df.Start, tmp_df.Stop, tmp_df.Color))
            tmp_df.Region_name=tmp_df.Region_name.apply(lambda x: '' if pd.isnull(x) else x)
            regions=list(tmp_df.Region_name)
            if color_flag:
                bar=np.ones([len(delta_ens),4])
                for coord in coords:
                    bar[(coord[0]-1):coord[1]]=list(colors.to_rgba(coords[2]))
            else:
                bar=np.zeros(len(delta_ens))
                for coord in coords:
                    bar[(coord[0]-1):coord[1]]=-i

            bar=np.array([bar]*max(int(len(delta_ens)/80),1))
            if color_flag:
                im=ax[i].imshow(bar, vmin=-8, vmax=0)
            else:
                im=ax[i].imshow(bar, cmap=cm, vmin=-8, vmax=0)
            ax[i].axes.get_yaxis().set_visible(False)
            if len(coords)>0:
                for dom, (start, end, _) in zip(regions,coords):
                    if dom:
                        ax[i].text((start+end)/2, 0.5*int(len(delta_ens)/100), dom, fontsize='large', horizontalalignment='center',verticalalignment='center')

            ax[i].set_title(RBP_uniprotID+'_'+t,fontsize='x-large',verticalalignment='bottom')
        if draw_ensemble_only == True:
            ax[len(types)].plot(range(len(delta_ens)),delta_ens, lw=1.5, c='k')
            ax[len(types)].set_xlim(0,len(delta_ens))
            ax[len(types)].axhline(y=0,ls='-', color='grey', lw=2)
            ax[len(types)].set_title(RBP_uniprotID+'-'+ens_delta_zscore_track_name, fontsize='x-large')
            #ax[len(types)].plot(delta_ens_sig_coords, np.zeros(len(delta_ens_sig_coords)), 'r.')
            # Fill positive peak areas with blue and negative with purple
            ax[len(types)].fill_between(range(len(delta_ens)), delta_ens, where=(np.array(delta_ens)>=0), interpolate=True, color='steelblue')
            ax[len(types)].fill_between(range(len(delta_ens)), delta_ens, where=(np.array(delta_ens)<0), interpolate=True, color='mediumorchid')
            # Set the background color to "plum" with 20% transparency
            ax[len(types)].set_facecolor((0.867, 0.627, 0.867, 0.2))
    
            peaks_coords1=occ_df[occ_df['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue']<=p_threshold1]['occluded_coord']
            peaks_coords2=occ_df[occ_df['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue']<=p_threshold2]['occluded_coord']
            print("sig peak regions (p<{}): {}".format(p_threshold1,', '.join(merge_peaks(list(peaks_coords1)))))
            print("sig peak regions (p<{}): {}".format(p_threshold2,', '.join(merge_peaks(list(peaks_coords2)))))
            #print(occ_df[occ_df['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue']<=p_threshold2][['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue','occluded_coord']])
            peaks_pos1=set([])
            peaks_pos2=set([])
            for coo1 in peaks_coords1:
                peaks_pos1.update(list(range(int(coo1.split('-')[0]),int(coo1.split('-')[1])+1)))
            for coo2 in peaks_coords2:
                peaks_pos2.update(list(range(int(coo2.split('-')[0]),int(coo2.split('-')[1])+1)))    
            peak_bar=np.zeros(len(occ_df)+19)
    #                 print(peaks_pos1)
    #                 print(peaks_pos2)
            for i in peaks_pos1:
                peak_bar[i]=1
            for j in peaks_pos2:
                peak_bar[j]=2
            peak_bar=np.array([peak_bar]*max(int(len(occ_df)/60),1))
            ax[len(types)+1].imshow(peak_bar, cmap=pvalue_cmap, vmin=0, vmax=2)
            ax[len(types)+1].set_title('Significant occlusion peaks',fontsize='x-large',verticalalignment='bottom')
            ## Create custom legend lines
            custom_lines = [Line2D([0], [0], color="lightskyblue", lw=6),
                            Line2D([0], [0], color="steelblue", lw=6)]
            # Add the legend to the same axes as the imshow plot
            ax[len(types)+1].legend(custom_lines, ['p < {}'.format(p_threshold1), 'p < {}'.format(p_threshold2)], loc='best', bbox_to_anchor=(1.05, -2), fontsize='xx-large')
            plt.tight_layout()
            if not run_fdr_ensemble:
                f.savefig(os.path.join(out_dir,RBP_uniprotID+'_OcclusionMap1D.pdf'),format='pdf')
                f.savefig(os.path.join(out_dir,RBP_uniprotID+'_OcclusionMap1D.png'),format='png')
            else:
                f.savefig(os.path.join(out_dir,RBP_uniprotID+'_FdrEns_OcclusionMap1D.pdf'),format='pdf')
                f.savefig(os.path.join(out_dir,RBP_uniprotID+'_FdrEns_OcclusionMap1D.png'),format='png')
        else:
            ax[len(types)].plot(range(len(delta_SVM)),delta_SVM, lw=2)
            ax[len(types)].set_xlim(0,len(delta_SVM))
            ax[len(types)].axhline(y=0,ls='-', color='grey', lw=2)
            ax[len(types)].set_title('Standardized Delta Scores in seqSVM, (Occ - origin) | orig_seqSVM: {}'.format(orig_SVM),fontsize='x-large')
            ax[len(types)].plot(delta_SVM_sig_coords, np.zeros(len(delta_SVM_sig_coords)), 'r.')
            ax[len(types)+1].plot(range(len(delta_DNN)),delta_DNN, lw=2)
            ax[len(types)+1].set_xlim(0,len(delta_DNN))
            ax[len(types)+1].axhline(y=0,ls='-', color='grey', lw=2)
            ax[len(types)+1].set_title('Standardized Delta Scores in seqCNN, (Occ - origin) | orig_seqCNN: {}'.format(orig_DNN),fontsize='x-large')
            ax[len(types)+1].plot(delta_DNN_sig_coords, np.zeros(len(delta_DNN_sig_coords)), 'r.')
            ax[len(types)+2].plot(range(len(delta_ProteinBERT)),delta_ProteinBERT, lw=2)
            ax[len(types)+2].set_xlim(0,len(delta_ProteinBERT))
            ax[len(types)+2].axhline(y=0,ls='-', color='grey', lw=2)
            ax[len(types)+2].set_title('Standardized Delta Scores in ProteinBERT, (Occ - origin) | orig_ProteinBERT: {}'.format(orig_ProteinBERT),fontsize='x-large')
            ax[len(types)+2].plot(delta_ProteinBERT_sig_coords, np.zeros(len(delta_ProteinBERT_sig_coords)), 'r.')
            ax[len(types)+3].plot(range(len(delta_ens)),delta_ens, lw=1.5, c='k')
            ax[len(types)+3].set_xlim(0,len(delta_ens))
            ax[len(types)+3].axhline(y=0,ls='-', color='grey', lw=2)
            ax[len(types)+3].set_title(RBP_uniprotID+'-'+ens_delta_zscore_track_name, fontsize='x-large')
            #ax[len(types)+3].plot(delta_ens_sig_coords, np.zeros(len(delta_ens_sig_coords)), 'r.')
            # Fill positive peak areas with blue and negative with purple
            ax[len(types)+3].fill_between(range(len(delta_ens)), delta_ens, where=(np.array(delta_ens)>=0), interpolate=True, color='steelblue')
            ax[len(types)+3].fill_between(range(len(delta_ens)), delta_ens, where=(np.array(delta_ens)<0), interpolate=True, color='mediumorchid')
            # Set the background color to "plum" with 20% transparency
            ax[len(types)+3].set_facecolor((0.867, 0.627, 0.867, 0.2))
    
            peaks_coords1=occ_df[occ_df['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue']<=p_threshold1]['occluded_coord']
            peaks_coords2=occ_df[occ_df['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue']<=p_threshold2]['occluded_coord']
            print("sig peak regions (p<{}): {}".format(p_threshold1,', '.join(merge_peaks(list(peaks_coords1)))))
            print("sig peak regions (p<{}): {}".format(p_threshold2,', '.join(merge_peaks(list(peaks_coords2)))))
            #print(occ_df[occ_df['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue']<=p_threshold2][['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue','occluded_coord']])
            peaks_pos1=set([])
            peaks_pos2=set([])
            for coo1 in peaks_coords1:
                peaks_pos1.update(list(range(int(coo1.split('-')[0]),int(coo1.split('-')[1])+1)))
            for coo2 in peaks_coords2:
                peaks_pos2.update(list(range(int(coo2.split('-')[0]),int(coo2.split('-')[1])+1)))    
            peak_bar=np.zeros(len(occ_df)+19)
    #                 print(peaks_pos1)
    #                 print(peaks_pos2)
            for i in peaks_pos1:
                peak_bar[i]=1
            for j in peaks_pos2:
                peak_bar[j]=2
            peak_bar=np.array([peak_bar]*max(int(len(occ_df)/60),1))
            ax[len(types)+4].imshow(peak_bar, cmap=pvalue_cmap, vmin=0, vmax=2)
            ax[len(types)+4].set_title('Significant occlusion peaks',fontsize='x-large',verticalalignment='bottom')
            ## Create custom legend lines
            custom_lines = [Line2D([0], [0], color="lightskyblue", lw=6),
                            Line2D([0], [0], color="steelblue", lw=6)]
            # Add the legend to the same axes as the imshow plot
            ax[len(types)+4].legend(custom_lines, ['p < {}'.format(p_threshold1), 'p < {}'.format(p_threshold2)], loc='best', bbox_to_anchor=(1.05, -2), fontsize='xx-large')
        
            plt.tight_layout()
            if not run_fdr_ensemble:
                f.savefig(os.path.join(out_dir,RBP_uniprotID+'_OcclusionMap1D.pdf'),format='pdf')
                f.savefig(os.path.join(out_dir,RBP_uniprotID+'_OcclusionMap1D.png'),format='png')
            else:
                f.savefig(os.path.join(out_dir,RBP_uniprotID+'_FdrEns_OcclusionMap1D.pdf'),format='pdf')
                f.savefig(os.path.join(out_dir,RBP_uniprotID+'_FdrEns_OcclusionMap1D.png'),format='png')
    else:
        if draw_ensemble_only == True:
            f, (ax4, ax5) = plt.subplots(2, 1, figsize=(20, 6))
        else:
            f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(20, 15)) 
            ax1.plot(range(len(delta_SVM)),delta_SVM, lw=2)
            ax1.set_xlim(0,len(delta_SVM))
            ax1.axhline(y=0,ls='-', color='grey', lw=2)
            ax1.set_title('Standardized Delta Scores in SVM, (Occ - origin) | orig_seqSVM: {}'.format(orig_SVM),fontsize='x-large')
            ax1.plot(delta_SVM_sig_coords, np.zeros(len(delta_SVM_sig_coords)), 'r.')
            ax2.plot(range(len(delta_DNN)),delta_DNN, lw=2)
            ax2.set_xlim(0,len(delta_DNN))
            ax2.axhline(y=0,ls='-', color='grey', lw=2)
            ax2.set_title('Standardized Delta Scores in DNN, (Occ - origin) | orig_seqCNN: {}'.format(orig_DNN),fontsize='x-large')
            ax2.plot(delta_DNN_sig_coords, np.zeros(len(delta_DNN_sig_coords)), 'r.')
            ax3.plot(range(len(delta_ProteinBERT)),delta_ProteinBERT, lw=2)
            ax3.set_xlim(0,len(delta_ProteinBERT))
            ax3.axhline(y=0,ls='-', color='grey', lw=2)
            ax3.set_title('Standardized Delta Scores in ProteinBERT, (Occ - origin) | orig_ProteinBERT: {}'.format(orig_ProteinBERT),fontsize='x-large')
            ax3.plot(delta_ProteinBERT_sig_coords, np.zeros(len(delta_ProteinBERT_sig_coords)), 'r.')
            
        ax4.plot(range(len(delta_ens)),delta_ens, lw=1.5, c='k')
        ax4.set_xlim(0,len(delta_ens))
        ax4.axhline(y=0,ls='-', color='grey', lw=2)
        ax4.set_title(RBP_uniprotID+'-'+ens_delta_zscore_track_name, fontsize='x-large')
        ax4.plot(delta_ens_sig_coords, np.zeros(len(delta_ens_sig_coords)), 'r.')
        ax4.fill_between(range(len(delta_ens)), delta_ens, where=(np.array(delta_ens)>=0), interpolate=True, color='steelblue')
        ax4.fill_between(range(len(delta_ens)), delta_ens, where=(np.array(delta_ens)<0), interpolate=True, color='mediumorchid')
        # Set the background color to "plum" with 20% transparency
        ax4.set_facecolor((0.867, 0.627, 0.867, 0.2))
    
        peaks_coords1=occ_df[occ_df['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue']<=p_threshold1]['occluded_coord']
        peaks_coords2=occ_df[occ_df['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue']<=p_threshold2]['occluded_coord']
        print('All the print out coordinates are 0-based (starting from 0 rather than 1).')
        print("sig peak regions (p<{}): {}".format(p_threshold1,', '.join(merge_peaks(list(peaks_coords1)))))
        print("sig peak regions (p<{}): {}".format(p_threshold2,', '.join(merge_peaks(list(peaks_coords2)))))
        #print(occ_df[occ_df['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue']<=p_threshold2][['avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT_pvalue','occluded_coord']])
        peaks_pos1=set([])
        peaks_pos2=set([])
        for coo1 in peaks_coords1:
            peaks_pos1.update(list(range(int(coo1.split('-')[0]),int(coo1.split('-')[1])+1)))
        for coo2 in peaks_coords2:
            peaks_pos2.update(list(range(int(coo2.split('-')[0]),int(coo2.split('-')[1])+1)))    
        peak_bar=np.zeros(len(occ_df)+19)
#                 print(peaks_pos1)
#                 print(peaks_pos2)
        for i in peaks_pos1:
            peak_bar[i]=1
        for j in peaks_pos2:
            peak_bar[j]=2
        peak_bar=np.array([peak_bar]*max(int(len(occ_df)/60),1))
        ax5.imshow(peak_bar, cmap=pvalue_cmap, vmin=0, vmax=2)
        ax5.set_title('Significant occlusion peaks',fontsize='x-large', verticalalignment='bottom')
        ## Create custom legend lines
        custom_lines = [Line2D([0], [0], color="lightskyblue", lw=6),
                        Line2D([0], [0], color="steelblue", lw=6)]
        # Add the legend to the same axes as the imshow plot
        ax5.legend(custom_lines, ['p < {}'.format(p_threshold1), 'p < {}'.format(p_threshold2)], loc='best', bbox_to_anchor=(1.05, -2), fontsize='xx-large')

        plt.tight_layout()
        if not run_fdr_ensemble:
            f.savefig(os.path.join(out_dir,RBP_uniprotID+'_OcclusionMap1D.pdf'),format='pdf')
            f.savefig(os.path.join(out_dir,RBP_uniprotID+'_OcclusionMap1D.png'),format='png')
        else:
            f.savefig(os.path.join(out_dir,RBP_uniprotID+'_FdrEns_OcclusionMap1D.pdf'),format='pdf')
            f.savefig(os.path.join(out_dir,RBP_uniprotID+'_FdrEns_OcclusionMap1D.png'),format='png')
        
def main(args):
    no_secondary_structure=args.no_secondary_structure
    k=args.window_size
    maxlen=args.maxlen
    max_seq_len=args.max_seq_len
    BioVec_weights_file=args.BioVec_weights
    seqDNN_modelfile_stru=args.seqDNN_modelfile_stru
    seqDNN_modelfile_weight=args.seqDNN_modelfile_weight
    seqSVM_modelfile=args.seqSVM_modelfile
    proteinBERT_modelfile=args.proteinBERT_modelfile
    seq_files=args.seq_files
    seq_dir=args.seq_dir
    ref_seq_dir=args.ref_seq_dir
    use_Zscore=args.use_Zscore
    annotation_file=args.annotation_file
    annotation_file_separator=args.annotation_file_separator
    print("use_Zscore = {}".format(str(use_Zscore)))
    if not seq_files:
        seq_files=list(map(lambda x: os.path.join(seq_dir,x), os.listdir(seq_dir)))
    else:
        seq_files=seq_files.split(',')
    out_dir=args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_dir2=args.out_dir2
    if out_dir2:
        Path(out_dir2).mkdir(parents=True, exist_ok=True)
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
    reference_all_score_file=args.reference_all_score_file
    peak_pval_threshold=args.peak_pval_threshold
    Model_name=args.model_name
    model_name=args.model_name
    model_dir=args.model_dir
    SVM_col=args.SVM_col
    DNN_col=args.DNN_col  
    ProteinBERT_col=args.ProteinBERT_col
    seq_score_col=args.seq_score_col
    n_annotations=args.n_annotations
    start_seq_len_ProteinBERT=args.start_seq_len_ProteinBERT
    run_fdr_ensemble=args.run_fdr_ensemble
    upper_bound_protlenNorm=args.upper_bound_protlenNorm
    plotting_occlusion=args.plotting_occlusion
    draw_ensemble_only=args.draw_ensemble_only

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
                print('LOOK HERE0.')
                # seqSVM_modelfile = pkg_resources.resource_stream(__name__, '../pre_trained/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_noSS_WithNoSSmodel_ModelFile.pkl')
                seqSVM_modelfile = pkg_resources.resource_stream(__name__, '../pre_trained/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_noSS_WithNoSSmodel_ModelFile.pickle.pkl')
                print('LOOK FORWARD0.')
        else:
            if seqDNN_modelfile_stru == None:
                seqDNN_modelfile_stru = pkg_resources.resource_string(__name__, '../pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_model_structure.json').decode("utf-8")
            else:
                with open(seqDNN_modelfile_stru) as f:
                    seqDNN_modelfile_stru=f.read()
            if seqDNN_modelfile_weight == None:
                seqDNN_modelfile_weight = pkg_resources.resource_filename(__name__, '../pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_model_weights.h5')
            if seqSVM_modelfile == None:
                # seqSVM_modelfile = pkg_resources.resource_stream(__name__, '../pre_trained/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_ModelFile.pkl')
                print('LOOK HERE.')
                seqSVM_modelfile = pkg_resources.resource_stream(__name__, '../pre_trained/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_ModelFile.pickle.pkl')
                print(seqSVM_modelfile)
                #model_SVM = pickle.load(seqSVM_modelfile)
                model_SVM=joblib.load(seqSVM_modelfile)
                print(model_SVM)
                print('LOOK FORWARD.')
                #print(model_SVM)
        if proteinBERT_modelfile==None:
            raise ValueError('--proteinBERT_modelfile argument is required.')
            # proteinBERT_modelfile = pkg_resources.resource_filename(__name__, '../pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_noSS_WithNoSSmodel_model_weights.h5')
            # proteinBERT_modelfile = pkg_resources.resource_stream(__name__, '../data/HydRa_v4_7_3_Occlusion_maps_seqSVM_seqDNN_scores_statistics.pkl')

    else:
        seqDNN_modelfile_stru=os.path.join(model_dir, Model_name+'_seqDNN_model_structure.json')
        with open(seqDNN_modelfile_stru) as f:
                seqDNN_modelfile_stru=f.read()

        seqDNN_modelfile_weight=os.path.join(model_dir, Model_name+'_seqDNN_model_weights.h5')
        seqSVM_modelfile = os.path.join(model_dir, Model_name+'_seqSVM_ModelFile.pkl')
        proteinBERT_modelfile = os.path.join(model_dir, Model_name+'_ProteinBERT_ModelFile.pkl')

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
            reference_score_file = pkg_resources.resource_stream(__name__, '../data/Classification_scores_reference_cv_HydRa2.0_DNN_SVM_ProteinBERT_menthaBioPlex_STRING_noSS.tsv')
        else:
            reference_score_file = pkg_resources.resource_stream(__name__, '../data/Classification_scores_reference_cv_HydRa2.0_DNN_SVM_ProteinBERT_menthaBioPlex_STRING.tsv')
    
    if reference_all_score_file == None:
        if no_secondary_structure:
            reference_all_score_file = pkg_resources.resource_stream(__name__, '../data/Classification_scores_TrainingWithTheWholeProteinSet_final_model_HydRa2.0_DNN_SVM_ProteinBERT_menthaBioPlex_STRING_noSS.tsv')
        else:
            reference_all_score_file = pkg_resources.resource_stream(__name__, '../data/Classification_scores_TrainingWithTheWholeProteinSet_final_model_HydRa2.0_DNN_SVM_ProteinBERT_menthaBioPlex_STRING.tsv')

        

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

    if use_Zscore==True and (ref_seq_dir==None or out_dir2==None):
        logging.info('Arguments for --ref_seq_dir or --out_dir_ref is not provided, the default seqSVM_seqDNN_score_stats from pre-trained model will be used.')
        #seqSVM_seqDNN_score_stats_file=pkg_resources.resource_stream(__name__, '../data/HydRa_v4_7_3_Occlusion_maps_seqSVM_seqDNN_scores_statistics.pkl')
        seqSVM_seqDNN_score_stats_file=pkg_resources.resource_stream(__name__, '../data/HydRa2.0_noSS_Occlusion_maps_seqSVM_seqDNN_ProteinBERT_proteinLengthWiseFib_scores_statistics.pkl')
        seqSVM_seqDNN_score_stats_dic=pickle.load(seqSVM_seqDNN_score_stats_file)
        
        #avg_zscore_stats_file=pkg_resources.resource_stream(__name__, '../data/HydRa_v4_7_3_Occlusion_maps_seqSVM_seqDNN_avg_zscores_statistics.pkl')
        avg_zscore_stats_file=pkg_resources.resource_stream(__name__, '../data/HydRa2.0_noSS_Occlusion_maps_seqSVM_seqDNN_ProteinBERT_proteinLengthWiseFib_avg_zscores_statistics.pkl')
        avg_zscore_stats_dic=pickle.load(avg_zscore_stats_file)

    ## For naive ensembling (probabilistic)
    AAs=['H','K','D','E','S','T','N','Q','C','G','P','A','V','I','L','M','F','Y','W','R']

    reference_score_df=pd.read_table(reference_score_file, index_col=0)

    scores_DNN=list(reference_score_df['seqDNN_score'])
    scores_SVM=list(reference_score_df['seqSVM_score'])
    scores_ProteinBERT=list(reference_score_df['seqSVM_score']) ## To be edited
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

    ###### Generate score statistics from reference dataset
    if use_Zscore==True and ref_seq_dir!=None and out_dir2!=None:
        ### Run zscore normalization, zscore integration and p-value calcuation for reference data.
        HydRa_score_whole_df=pd.read_table(reference_all_score_file, index_col=0)
        if not ("Protein_length" in HydRa_score_whole_df.columns):
            HydRa_score_whole_df['Protein_length']=list(map(lambda x: get_protein_length(x, ref_seq_dir), HydRa_score_whole_df.index))

        prot_len_dic0=HydRa_score_whole_df['Protein_length'].to_dict()
        ## get reference score dict for deltaSVM and deltaDNN
        cutoff=1.0
        scores_dic={}
        scores_dic[SVM_col]={}
        scores_dic[DNN_col]={}
        scores_dic[ProteinBERT_col]={}
        scores_dic[seq_score_col]={}
        start_len=100
        start_len0=0
        ## The sequence interval goes in Fibonacci sequence.
        while start_len0<=upper_bound_protlenNorm:
            if start_len0==start_len:
                start_len2=start_len
                start_len=start_len+start_len0
                start_len0=start_len2
                continue
            if start_len>upper_bound_protlenNorm:
                prot_set=list(HydRa_score_whole_df[HydRa_score_whole_df["Protein_length"]>start_len0].index)
                start_len=np.inf
            else:
                prot_set=list(HydRa_score_whole_df[(HydRa_score_whole_df["Protein_length"]>start_len0) & (HydRa_score_whole_df["Protein_length"]<=start_len)].index)
            
            prot_set=[x for x in prot_set if os.path.exists(os.path.join(ref_seq_dir, x+'.fasta'))]
            #prot_set=[x for x in prot_set if os.path.exists(os.path.join(out_dir2, x+'_Occlusion_score_matrix_aac.xls'))]
            shuffle(prot_set)
            scores_dic[SVM_col]["{}-{}".format(start_len0,start_len)]=[]
            scores_dic[DNN_col]["{}-{}".format(start_len0,start_len)]=[]
            scores_dic[ProteinBERT_col]["{}-{}".format(start_len0,start_len)]=[]
            scores_dic[seq_score_col]["{}-{}".format(start_len0,start_len)]=[]
                        
            if run_fdr_ensemble:
                scores_dic[seq_score_col]["{}-{}".format(start_len0,start_len)]=[]

            prot_set_to_occlude=[p for p in prot_set if not os.path.exists(os.path.join(out_dir2, p+'_Occlusion_score_matrix_aac.xls'))]
            if no_secondary_structure:
                if len(prot_set_to_occlude)>0:
                    wrap_run_occlusion_HydRa2(prot_set_to_occlude, ref_seq_dir, out_dir2, selected_aa3mers_file, selected_aa4mers_file, selected_AAC_file, combined_selected_feature, autoEncoder_CNN_file, autoEncoder_Dense_file, BioVec_weights_add_null, maxlen, max_seq_len, seqDNN_modelfile_stru, seqDNN_modelfile_weight, seqSVM_modelfile, proteinBERT_modelfile, model_dir, model_name, no_secondary_structure, k, BioVec_name_dict, AAs, scores_DNN, scores_SVM, true_labels, n_annotations=n_annotations, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT)            
                if run_fdr_ensemble:
                    if len(prot_set_to_occlude)>0:
                        wrap_run_occlusion_HydRa2(prot_set_to_occlude, ref_seq_dir, out_dir2, selected_aa3mers_file, selected_aa4mers_file, selected_AAC_file, combined_selected_feature, autoEncoder_CNN_file, autoEncoder_Dense_file, BioVec_weights_add_null, maxlen, max_seq_len, seqDNN_modelfile_stru, seqDNN_modelfile_weight, seqSVM_modelfile, proteinBERT_modelfile, model_dir, model_name, no_secondary_structure, k, BioVec_name_dict, AAs, scores_DNN, scores_SVM, true_labels, n_annotations=n_annotations, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT)            
            else:
                if len(prot_set_to_occlude)>0:
                    wrap_run_occlusion_HydRa2(prot_set_to_occlude, ref_seq_dir, out_dir2, selected_aa3mers_file, selected_aa4mers_file, selected_AAC_file, combined_selected_feature, autoEncoder_CNN_file, autoEncoder_Dense_file, BioVec_weights_add_null, maxlen, max_seq_len, seqDNN_modelfile_stru, seqDNN_modelfile_weight, seqSVM_modelfile, proteinBERT_modelfile, model_dir, model_name, no_secondary_structure, k, BioVec_name_dict, AAs, scores_DNN, scores_SVM, true_labels, selected_SS11mers_file=selected_SS11mers_file, selected_SS15mers_file=selected_SS15mers_file, n_annotations=n_annotations, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT)
                if run_fdr_ensemble:
                    if len(prot_set_to_occlude)>0:
                        wrap_run_occlusion_HydRa2(prot_set_to_occlude, ref_seq_dir, out_dir2, selected_aa3mers_file, selected_aa4mers_file, selected_AAC_file, combined_selected_feature, autoEncoder_CNN_file, autoEncoder_Dense_file, BioVec_weights_add_null, maxlen, max_seq_len, seqDNN_modelfile_stru, seqDNN_modelfile_weight, seqSVM_modelfile, proteinBERT_modelfile, model_dir, model_name, no_secondary_structure, k, BioVec_name_dict, AAs, scores_DNN, scores_SVM, true_labels, selected_SS11mers_file=selected_SS11mers_file, selected_SS15mers_file=selected_SS15mers_file, n_annotations=n_annotations, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT)

            for prot in prot_set:
                t_df=pd.read_table(os.path.join(out_dir2, prot+'_Occlusion_score_matrix_aac.xls'))
                scores_dic[DNN_col]["{}-{}".format(start_len0,start_len)]+=list(t_df['delta_DNN'])
                scores_dic[SVM_col]["{}-{}".format(start_len0,start_len)]+=list(t_df['delta_SVM'])
                scores_dic[ProteinBERT_col]["{}-{}".format(start_len0,start_len)]+=list(t_df['delta_ProteinBERT'])

            if run_fdr_ensemble:
                wrap_run_ensemble(prot_set, out_dir2, scores_DNN=scores_DNN, scores_SVM=scores_SVM, scores_ProteinBERT=scores_ProteinBERT, true_labels=true_labels)
                for prot in prot_set:
                    t_df=pd.read_table(os.path.join(out_dir2, prot+'_Occlusion_score_matrix_full_aac_fdrEnsemble.xls'))
                    scores_dic[seq_score_col]["{}-{}".format(start_len0,start_len)]+=list(t_df['delta_fdr_ens'])

            start_len2=start_len
            #start_len0=start_len
            if start_len<=upper_bound_protlenNorm:
                #start_len+=start_len
                start_len=start_len+start_len0

            start_len0=start_len2
            
        scores_dic[DNN_col]['all']=np.concatenate(list(scores_dic[DNN_col].values()))
        scores_dic[SVM_col]['all']=np.concatenate(list(scores_dic[SVM_col].values()))
        scores_dic[ProteinBERT_col]['all']=np.concatenate(list(scores_dic[ProteinBERT_col].values()))
        if run_fdr_ensemble:
            scores_dic[seq_score_col]['all']=np.concatenate(list(scores_dic[seq_score_col].values()))

        seqSVM_seqDNN_score_stats_dic=get_stats_dic(scores_dic)

        with open(os.path.join(out_dir2,model_name+'_Occlusion_maps_seqSVM_seqDNN_ProteinBERT_proteinLengthWiseFib_scores_statistics.pkl'), 'wb') as f:
            pickle.dump(seqSVM_seqDNN_score_stats_dic, f)

        ## generate
        avg_zscore_stats_dic=get_avg_zscore_stats(HydRa_score_whole_df, seqSVM_seqDNN_score_stats_dic, out_dir2, out_dir2, SVM_col, DNN_col, ProteinBERT_col, prot_len_dic0)
        with open(os.path.join(out_dir2,model_name+'_Occlusion_maps_seqSVM_seqDNN_ProteinBERT_proteinLengthWiseFib_avg_zscores_statistics.pkl'), 'wb') as f:
            pickle.dump(avg_zscore_stats_dic, f)
        if not run_fdr_ensemble:
            prepare_z_score(avg_zscore_stats_dic, out_dir2, avg_col='avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT', peak_pval_threshold=peak_pval_threshold)
        else:
            prepare_z_score_for_fdrEnsemble(out_dir2, seqSVM_seqDNN_score_stats_dic, seq_score_col, delta_ens_col='delta_fdr_ens', ens_delta_zscore_col='zscore_deltaFdrEns',peak_pval_threshold=peak_pval_threshold)


    ###### Run zscore normalization, zscore integration and p-value calcuation for the query data.
    RBP_files=[f for f in seq_files if f.endswith('.fasta') and os.path.exists(f)]
    RBPs=['.'.join(f.strip(' ').split('/')[-1].split('.')[:-1]) for f in RBP_files]
    shuffle(RBPs)

    if no_secondary_structure:
        wrap_run_occlusion_HydRa2([p for p in RBPs if not os.path.exists(os.path.join(out_dir, p+'_Occlusion_score_matrix_aac.xls'))], None, out_dir, selected_aa3mers_file, selected_aa4mers_file, selected_AAC_file, combined_selected_feature, autoEncoder_CNN_file, autoEncoder_Dense_file, BioVec_weights_add_null, maxlen, max_seq_len, seqDNN_modelfile_stru, seqDNN_modelfile_weight, seqSVM_modelfile, proteinBERT_modelfile, model_dir, model_name, no_secondary_structure, k, BioVec_name_dict, AAs, scores_DNN, scores_SVM, true_labels, n_annotations=n_annotations, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT, seq_files=RBP_files)                
    else:
        wrap_run_occlusion_HydRa2([p for p in RBPs if not os.path.exists(os.path.join(out_dir, p+'_Occlusion_score_matrix_aac.xls'))], None, out_dir, selected_aa3mers_file, selected_aa4mers_file, selected_AAC_file, combined_selected_feature, autoEncoder_CNN_file, autoEncoder_Dense_file, BioVec_weights_add_null, maxlen, max_seq_len, seqDNN_modelfile_stru, seqDNN_modelfile_weight, seqSVM_modelfile, proteinBERT_modelfile, model_dir, model_name, no_secondary_structure, k, BioVec_name_dict, AAs, scores_DNN, scores_SVM, true_labels, selected_SS11mers_file=selected_SS11mers_file, selected_SS15mers_file=selected_SS15mers_file, n_annotations=n_annotations, start_seq_len_ProteinBERT=start_seq_len_ProteinBERT, seq_files=RBP_files)            
    
    if run_fdr_ensemble:
        wrap_run_ensemble(prot_set_FdrEns, out_dir, scores_DNN=scores_DNN, scores_SVM=scores_SVM, scores_ProteinBERT=scores_ProteinBERT, true_labels=true_labels)

    prot_len_dic={'.'.join(f.strip(' ').split('/')[-1].split('.')[:-1]):get_protein_length(f) for f in RBP_files}

    if use_Zscore==True:
        if (ref_seq_dir==None) or (out_dir2==None):
            get_zscore(seqSVM_seqDNN_score_stats_dic, out_dir2, out_dir, SVM_col, DNN_col, ProteinBERT_col, prot_len_dic)
        else:
            avg_zscore_stats_dic=get_avg_zscore_stats(HydRa_score_whole_df, seqSVM_seqDNN_score_stats_dic, out_dir2, out_dir, SVM_col, DNN_col, ProteinBERT_col, prot_len_dic)

        prepare_z_score(avg_zscore_stats_dic, out_dir, avg_col='avg_zscore_deltaSVM_deltaDNN_deltaProteinBERT', peak_pval_threshold=peak_pval_threshold)
        if run_fdr_ensemble:
            prepare_z_score_for_fdrEnsemble(out_dir, seqSVM_seqDNN_score_stats_dic, seq_score_col, delta_ens_col='delta_fdr_ens', ens_delta_zscore_col='zscore_deltaFdrEns',peak_pval_threshold=peak_pval_threshold)

        # run_ensemble(RBP, out_dir, scores_DNN=scores_DNN, scores_SVM=scores_SVM, true_labels=true_labels)
        if plotting_occlusion:
            for RBP in RBPs:
                if run_fdr_ensemble:
                    plot_occlusion(RBP, out_dir, k, annotation_file=annotation_file,annotation_file_separator=annotation_file_separator, ens_delta_zscore_col='zscore_deltaFdrEns', ens_delta_zscore_track_name='Zscore of Delta FDR ensemble score', run_fdr_ensemble=True, draw_ensemble_only=draw_ensemble_only)
                else:
                    plot_occlusion(RBP, out_dir, k, annotation_file=annotation_file,annotation_file_separator=annotation_file_separator, draw_ensemble_only=draw_ensemble_only)


    #pool=mp.Pool(processes=16)
    #result=[pool.apply_async(run, args=(RBP, model_SeqOnly, k,)) for RBP in RBPs]
    #results=[p.get() for p in result]


def call_main():
    usage="""\nocclusion_map -s input_dir -f fasta_files """ 
    description="""Use trained HydRa to predict the RNA-binding capacity of given proteins. ps: If at least one of the -p/--PPI_feature_file or -P/--PPI2_feature_file is provided, then PPI_edgelist and PAI_edgelist will be ignored."""
    parser= ArgumentParser(usage=usage, description=description)
    #parser.add_option("-h", "--help", action="help")
    parser.add_argument('-w', '--window_size', dest='window_size', type=int ,default=20)
    parser.add_argument('-m', '--maxlen', dest='maxlen', help='', type=int, default=1500)
    parser.add_argument('-M', '--max_seq_len', dest='max_seq_len', help='', type=int, default=1500)
    parser.add_argument('--n_annotations', dest='n_annotations', help='The number of annotation dimension in the pre-trained ProteinBERT model.', type=int, default=8943)
    parser.add_argument('--start_seq_len_ProteinBERT', dest='start_seq_len_ProteinBERT', help='The starting protein sequence length that are used to create the input window for ProteinBERT model.', type=int, default=512)
    parser.add_argument('-b', '--BioVec_weights', dest='BioVec_weights', help='', metavar='FILE', default=None)
    parser.add_argument('-f', '--seq_files', dest='seq_files', help='fasta files of the given proteins.', type=str, default=None)
    parser.add_argument('-s', '--seq_dir', dest='seq_dir', help='The directory for processed sequence and secondary structure files. The sequence files should have suffix .fasta and the secondary structure with suffix .spd3 or .txt.', type=str, default=None)
    parser.add_argument('-R', '--ref_seq_dir', dest='ref_seq_dir', help='The directory for processed sequence and secondary structure files of the training set used in HydRa training. The sequence files should have suffix .fasta and the secondary structure with suffix .spd3 or .txt.', type=str, default=None)
    parser.add_argument('--not_use_Zscore', dest='use_Zscore', help='Standardize the occlusion delta scores and calling peaks based on the standardized scores (z-scores).', action='store_false')
    parser.add_argument('-o', '--out_dir', dest='out_dir', help='directory for output Occlusion Map files. Default: ./OcclusionMap_out.', type=str, default='./OcclusionMap_out')
    parser.add_argument('-O', '--out_dir_ref', dest='out_dir2', help='directory for output Occlusion Map files. Default: None', type=str, default=None)
    parser.add_argument('-D', '--seqDNN_modelfile_stru', dest='seqDNN_modelfile_stru', help='', metavar='FILE', default=None)
    parser.add_argument('-d', '--seqDNN_modelfile_weight', dest='seqDNN_modelfile_weight', help='', metavar='FILE', default=None)
    parser.add_argument('-S', '--seqSVM_modelfile', dest='seqSVM_modelfile', help='', metavar='FILE', default=None)
    parser.add_argument('-B', '--proteinBERT_modelfile', dest='proteinBERT_modelfile', help='', metavar='FILE', default=None)
    parser.add_argument('-r', '--reference_score_file', dest='reference_score_file', help='The file path of the reference score file from cross validation or final training .', metavar='STRING', default=None)
    parser.add_argument('-A', '--reference_all_score_file', dest='reference_all_score_file', help='The file path of the reference score file from final training.', metavar='STRING', default=None)
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
    parser.add_argument('-p', '--peak_pval_threshold', dest='peak_pval_threshold', help='The threshold for p values used in significant occlusion peaks calling.', type=float, default=0.05)
    parser.add_argument('--SVM_col', dest='SVM_col', help='Column name for SVM score', type=str, default='seqSVM_score')
    parser.add_argument('--DNN_col', dest='DNN_col', help='Column name for DNN score', type=str, default='seqDNN_score')
    parser.add_argument('--ProteinBERT_col', dest='ProteinBERT_col', help='Column name for ProteinBERT score', type=str, default='ProteinBERT_score')
    parser.add_argument('--seq_score_col', dest='seq_score_col', help='Column name for sequence-based ensemble score', type=str, default='optimistic_DNNSVMnoSSProteinBERT')
    parser.add_argument('--run_fdr_ensemble', dest='run_fdr_ensemble', help='Use FDR ensemble delta score as the final occlusion delta score instead of avg zscores', action='store_true')
    parser.add_argument('--secondary-structure', dest='no_secondary_structure', help='Use secondary structure information in the model training.', action='store_false')
    parser.add_argument('--upper_bound_protlenNorm', dest='upper_bound_protlenNorm', help='The upper bound of protein length when we normalize the occlusion scores groupped by protein length. So all the proteins with length larger than this value will form a single group and do the normalization (z-score transformation). Usually pick a number*100 in Fibonacci sequence.', type=int, default=2100)
    parser.add_argument('--not_plot_occlusion', dest='plotting_occlusion', help='Not plot the occlusion maps.', action='store_false')
    parser.add_argument('--draw_ensemble_only', dest='draw_ensemble_only', help='Only plot the occlusion maps from the ensemble predictions (which combine the outputs from seqSVM, seqCNN and ProteinBERT-RBP).', action='store_true')
    
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



