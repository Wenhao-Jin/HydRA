#!/home/wjin/envs/wjin/bin/python

import pandas as pd
import numpy as np
import multiprocessing as mp
import os
#from propy import PyPro
AAs=['H','K','D','E','S','T','N','Q','C','G','P','A','V','I','L','M','F','Y','W','R']

def count_selected_aakmers(s, selected_kmers):
    tmp_kmers_dict={kmer:0 for kmer in selected_kmers}
    #k=len(next(iter(selected_kmers)))
    if len(selected_kmers)==0:
        return {}
    k=len(selected_kmers[0])
    for i in range(len(s)-k+1):
        if (s[i:i+k] in selected_kmers):
            tmp_kmers_dict[s[i:i+k]]+=1
        
    return tmp_kmers_dict

def count_selected_SSkmers(s, selected_kmers):
    tmp_kmers_dict={kmer:0 for kmer in selected_kmers}
    #k=len(next(iter(selected_kmers)))
    if len(selected_kmers)==0:
        return {}
    k=len(selected_kmers[0])
    for i in range(len(s)-k+1):
        if (s[i:i+k] in selected_kmers):
            tmp_kmers_dict[s[i:i+k]]=tmp_kmers_dict[s[i:i+k]]+1
        
    return tmp_kmers_dict

def get_PseAAC_features(s):
    """
    return a dictionary of PAAC values.
    """
    s=s.replace('*','')
    s=s.replace('X','') #solve bug for those cannot be processed by propy
    s=s.replace('U','C') #solve bug for those 'U' that cannot be processed by propy
    try:
        DesObject=PyPro.GetProDes(s)
        paac=DesObject.GetPAAC(lamda=10, weight=0.05)  #the output is a dictionary
        #paac=DesObject.GetPAAC(lamda=0, weight=0.05)  #generate only 20 classic amino acid composition #the output is a dictionary 
        # paac['Protein_Name']=prot_file.split('/')[-1].split('.')[0]
    except Exception as e:
        print(type(e))
        print(e)
        # print prot_file.split('/')[-1].split('.')[0]
        return None

    return paac

def get_AAC_features(seq):
    dic = {aa:0 for aa in AAs}
    seq_len=len(seq)
    for aa in seq:
        if aa not in dic.keys():
            dic[aa]=1
        else:
            dic[aa]+=1

    return {k:v*1.0/seq_len for k, v in dic.items()}

def get_AAC_features2(prot_file, k=4):
    f=open(prot_file, 'r')
    s=''.join(f.read().strip('* \n').split('\n')[1:])
    f.close()
    s=s.upper()
    s=s.replace('*','')
    s=s.replace('X','') #solve bug for those cannot be processed by propy
    s=s.replace('U','C') #solve bug for those 'U' that cannot be processed by propy
    try:
        aac=get_AAC_features(s)  #the output is a dictionary
        aac['Protein_Name']=prot_file.split('/')[-1].split('.')[0]
    except Exception as e:
        print(type(e))
        print(e)
        print(prot_file.split('/')[-1].split('.')[0])
        return None

    return aac

def get_AAC_features_bk(s):
    """
    return a dictionary of PAAC values.
    """
    s=s.replace('*','')
    s=s.replace('X','') #solve bug for those cannot be processed by propy
    s=s.replace('U','C') #solve bug for those 'U' that cannot be processed by propy
    try:
        DesObject=PyPro.GetProDes(s)
        #paac=DesObject.GetPAAC(lamda=10, weight=0.05)  #the output is a dictionary
        paac=DesObject.GetPAAC(lamda=0, weight=0.05)  #generate only 20 classic amino acid composition #the output is a dictionary 
        # paac['Protein_Name']=prot_file.split('/')[-1].split('.')[0]
    except Exception as e:
        print(type(e))
        print(e)
        # print prot_file.split('/')[-1].split('.')[0]
        return None

    return paac

def Get_feature_table(seq, ss_seq, selected_aa3mers, selected_aa4mers, selected_SS11mers, selected_SS15mers, selected_AAC, combined_selected_feature, outfmt='vector'):
    """
    PseAAC: Whether to calculate Pseudo AAC. If False, only calculate classic amino acid composition.
    outfmt: str, can be chosen from 'vector', 'dict' and 'both'.
    """
    aa3mer_dic = count_selected_aakmers(seq, selected_aa3mers)
    aa4mer_dic = count_selected_aakmers(seq, selected_aa4mers)
    ss11mer_dic = count_selected_SSkmers(ss_seq, selected_SS11mers)
    ss15mer_dic = count_selected_SSkmers(ss_seq, selected_SS15mers)
    aac_dic = get_AAC_features(seq)
    aac_dic_selected = {k: aac_dic[k] for k in selected_AAC}

    aa3mer_dic.update(aa4mer_dic)
    aa3mer_dic.update(ss11mer_dic)
    aa3mer_dic.update(ss15mer_dic)
    aa3mer_dic.update(aac_dic_selected)

    print(combined_selected_feature[-30:])
    print(ss15mer_dic.keys()) 

    if outfmt=='vector' or outfmt=='both':
        ## To put the feature values in the order as SONAR+ defined. 
        #feature_ind = pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_SeqFeature_table_comined_all_selected_features_From_WholeDataset_12Apr_aac.txt',index_col=0).drop('RBP_flag',axis=1).columns
        #feature_ind = pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_SeqFeature_table_comined_all_selected_features_From_WholeDataset_12Apr_AACname.txt',index_col=0).drop('RBP_flag',axis=1).columns
        feature_ind = combined_selected_feature

        feature_vector = np.array([aa3mer_dic[k] for k in feature_ind])
        if outfmt=='vector':
            return feature_vector
        else:
            return feature_vector, aa3mer_dic

    elif outfmt=='dict':
        return aa3mer_dic
    else:
        raise ValueError('Invalid value of outfmt.')


def Get_feature_table_noSS(seq, selected_aa3mers, selected_aa4mers, selected_AAC, combined_selected_feature, outfmt='vector'):
    """
    PseAAC: Whether to calculate Pseudo AAC. If False, only calculate classic amino acid composition.
    outfmt: str, can be chosen from 'vector', 'dict' and 'both'.
    """
    aa3mer_dic = count_selected_aakmers(seq, selected_aa3mers)
    aa4mer_dic = count_selected_aakmers(seq, selected_aa4mers)
    aac_dic = get_AAC_features(seq)
    aac_dic_selected = {k: aac_dic[k] for k in selected_AAC}

    aa3mer_dic.update(aa4mer_dic)
    aa3mer_dic.update(aac_dic_selected)

    if outfmt=='vector' or outfmt=='both':
        ## To put the feature values in the order as SONAR+ defined. 
        #feature_ind = pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_SeqFeature_table_comined_all_selected_features_From_WholeDataset_12Apr_aac.txt',index_col=0).drop('RBP_flag',axis=1).columns
        #feature_ind = pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_SeqFeature_table_comined_all_selected_features_From_WholeDataset_12Apr_AACname.txt',index_col=0).drop('RBP_flag',axis=1).columns
        feature_ind = combined_selected_feature

        feature_vector = np.array([aa3mer_dic[k] for k in feature_ind])
        if outfmt=='vector':
            return feature_vector
        else:
            return feature_vector, aa3mer_dic

    elif outfmt=='dict':
        return aa3mer_dic
    else:
        raise ValueError('Invalid value of outfmt.')



