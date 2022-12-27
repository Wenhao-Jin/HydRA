#!/usr/bin/env python
import numpy as np
import os
from argparse import ArgumentParser
import pandas as pd
import math
from sklearn.metrics import roc_curve, auc, recall_score, precision_recall_curve, average_precision_score
from keras.models import load_model
#import matplotlib.pyplot as plt
from sklearn.svm import SVC
import joblib
from tensorflow.python.client import device_lib
from pathlib import Path
import multiprocessing as mp
from ..models.DNN_seq import SONARp_DNN_SeqOnly, SONARp_DNN_SeqOnly_2, SONARp_DNN_SeqOnly_noSS
from ..models.Sequence_class import Protein_Sequence_Input5, Protein_Sequence_Input5_2, Protein_Sequence_Input5_noSS
from ..preprocessing.Feature_generation.kmer_table_generation import Get_kmer_feature_table
from ..preprocessing.Feature_generation.AAC_feature_table_generation import get_AAC_features
from ..preprocessing.Feature_generation.PPI_feature_generation import get_PPI_features, get_1stPPI_features
from ..preprocessing.Feature_selection.Feature_selection_for_each_feature_after_chi2Filtering_forFinalTraining import chi2_filtering_followedBy_SVMselection, chi2_filtering_Only 
import pkg_resources
import warnings
import networkx as nx
#plt.switch_backend('agg')


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


def main(args):
    ## Load arguments from the command line.
    no_secondary_structure=args.no_secondary_structure
    maxlen=args.maxlen # Used in DNN and SVM_seq's model construction.
    len_file=args.len_file
    RBP_list=args.RBP_list
    BioVec_weights_file=args.BioVec_weights
    seq_dir=args.seq_dir
    train_list=args.train_list
    process_dir=args.process_dir
    Path(process_dir).mkdir(parents=True, exist_ok=True)
    union_file=args.seqSVM_ft_file  ## Feature table for seqSVM
    PPI_edgelist=args.PPI_edgelist
    PPA_edgelist=args.PPA_edgelist
    no_PIA=args.no_PIA
    no_PPA=args.no_PPA
    # PPI_feature_file=args.PPI_feature_file   # PPI feature table from MB
    # PPI2_feature_file=args.PPI2_feature_file  # PPI feature table from MB-STRING
    # reference_score_files=args.reference_score_files 
    new_pretrain=args.new_pretrain
    no_pretrain=args.no_pretrain
    autoEncoder_CNN_file=args.autoEncoder_CNN_file
    autoEncoder_Dense_file=args.autoEncoder_Dense_file
    Model_name=args.model_name
    model_outdir=args.model_outdir
    Path(model_outdir).mkdir(parents=True, exist_ok=True)
    alpha=args.alpha
    # uniprotID_mapping_file=args.uniprotID_mapping_file
    # score_outdir=args.score_outdir

    ## Load parameters data used for the model training.
    if RBP_list == None:
        RBP_list = pkg_resources.resource_string(__name__, '../data/Combined8RBPlist_plusOOPSXRNAXsharedRBPs_uniprotID_20190717.txt').decode("utf-8")
    else:
        with open(RBP_list) as f:
            RBP_list=f.read()
    if BioVec_weights_file == None:
        BioVec_weights_file = pkg_resources.resource_stream(__name__, '../data/protVec_100d_3grams.csv')
    if no_PIA==False:
        if PPI_edgelist == None:
            PPI_edgelist = pkg_resources.resource_stream(__name__, '../data/mentha20180108_BioPlex2.0_edgelist.txt')
        if no_PPA==False:
            if PPA_edgelist == None:
                PPA_edgelist = pkg_resources.resource_stream(__name__, '../data/STRING_v10.5_uniprot_edgelist_withoutExperimentalData.txt')
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

    print('Welcome to HydRa!')
    print('STEP 1: Preprocessing. Collecting proteins for training.')
    RBPs=set(RBP_list.split('\n'))

    ## Load the protein names of the training proteins. Only keep the proteins whose sequence (and secondary structure) information is available.
    if train_list == None:
        prots=list(map(lambda x: '.'.join(x.split('.')[:-1]), list(filter(lambda x: x.endswith('.fasta'), os.listdir(seq_dir)))))
    else:
        with open(train_list) as f:
            prots=f.read().split('\n')
            prots=list(filter(lambda x: os.path.exists(os.path.join(seq_dir,x+'.fasta')), prots))
 
    if no_secondary_structure==False:
        if train_list == None:
            prots2=set((map(lambda x: '.'.join(x.split('.')[:-1]), list(filter(lambda x: x.endswith('.txt') or x.endswith('.spd3'), os.listdir(seq_dir))))))
            prots=[p for p in prots if p in prots2]
        else:
            prots=list(filter(lambda x: os.path.exists(os.path.join(seq_dir,x+'.txt')) or os.path.exists(os.path.join(seq_dir,x+'.spd3')), prots))
        


    prot_seq_list=[]
    for prot in prots: 
        with open(os.path.join(seq_dir,prot+'.fasta')) as f:
            seq=''.join(f.read().strip(' \n').split('\n')[1:])
            prot_seq_list.append((prot, len(seq), prot in RBPs))

    Info_df=pd.DataFrame(prot_seq_list, columns=['Protein', 'protein_len','RBP_flag']).set_index('Protein')
    Info_df['RBP_flag']=Info_df['RBP_flag'].apply(lambda x: 1 if x else 0)

    print('STEP 1: Preprocessing. Prepare input data for seqDNN.')

    ## Get ID/names of the short proteins (no longer than the DNN windowsize, and will be used to train the seqDNN)
    prot_list_seq=list(set(Info_df[Info_df.protein_len<=maxlen].index))
    if no_secondary_structure==False:
        prot_list_seq=list(filter(lambda x: os.path.exists(os.path.join(seq_dir, x+'.spd3')) or os.path.exists(os.path.join(seq_dir, x+'.txt')),prot_list_seq))
    
    Info_df1 = Info_df.loc[prot_list_seq]

    ## Get ID/names of the long proteins (longer than the DNN windowsize, and will not be used in training)
    prot_list_seq2=list(set(Info_df[Info_df.protein_len>maxlen].index))
    if no_secondary_structure==False:
        prot_list_seq2=list(filter(lambda x: os.path.exists(os.path.join(seq_dir, x+'.spd3')) or os.path.exists(os.path.join(seq_dir, x+'.txt')),prot_list_seq2))

    Info_df2 = Info_df.loc[prot_list_seq2]  # dataframe for long proteins.


    ## For DNN
    BioVec_weights=pd.read_table(BioVec_weights_file, sep='\t', header=None, index_col=0)
    BioVec_weights_add_null=np.append(np.zeros((1,100)), BioVec_weights.values, axis=0) #append a [0,0,...,0] array at the top of the matrix, which used for padding 0s.
    BioVec_weights_add_null=BioVec_weights_add_null*10
    BioVec_name_dict={}
    for i in range(1, len(BioVec_weights)+1):
        BioVec_name_dict.update({BioVec_weights.index[i-1]:i})

    ### Data preparation (Training & Test).
    ## For DNN
    short_files = np.array([os.path.join(seq_dir, prot+'.fasta') for prot in Info_df1.index])
    class_labels_short = np.array(list(Info_df1.RBP_flag))
    Info_df3=Info_df1[Info_df1.RBP_flag==1]
    short_files_rbp = np.array([os.path.join(seq_dir, prot+'.fasta') for prot in Info_df3.index]) ## for seqDNN pre-training. Short RBPs.
    class_labels_short_rbp = np.array(list(Info_df3.RBP_flag)) ## for seqDNN pre-training. Short RBPs.
    if no_secondary_structure==True:
        dataset = Protein_Sequence_Input5_noSS(short_files, class_labels_short, BioVec_name_dict, maxlen=maxlen)
    else:
        dataset = Protein_Sequence_Input5(short_files, class_labels_short, BioVec_name_dict, maxlen=maxlen)
        X_ss_sparse_short = dataset.get_ss_sparse_mats2()

    X_aa3mer_short = dataset.get_aa3mer_mats()
    X_seqlens_short = dataset.get_seqlens()
    y_dnn_short = dataset.get_class_labels()
    usable_files_short = dataset.get_usable_files()
    prot_name_short = np.array(list(map(lambda x:x.split('/')[-1].split('.')[0], usable_files_short)))

    long_files = np.array([os.path.join(seq_dir, prot+'.fasta') for prot in Info_df2.index])
    class_labels_long = np.array(list(Info_df2.RBP_flag))
    if no_secondary_structure==True:
        dataset2 = Protein_Sequence_Input5_noSS(long_files, class_labels_long, BioVec_name_dict)
    else:
        dataset2 = Protein_Sequence_Input5(long_files, class_labels_long, BioVec_name_dict)
        X_ss_sparse_long = dataset2.get_ss_sparse_mats2()

    X_aa3mer_long = dataset2.get_aa3mer_mats()
    X_seqlens_long = dataset2.get_seqlens()
    y_dnn_long = dataset2.get_class_labels()
    usable_files_long = dataset2.get_usable_files()
    prot_name_long = np.array(list(map(lambda x:x.split('/')[-1].split('.')[0], usable_files_long)))
    max_seq_len = dataset2.get_maxlen()


    # X_aa3mer_mat = np.concatenate((X_aa3mer_short, X_aa3mer_long))
    # X_ss_sparse_mat = np.concatenate((X_ss_sparse_short, X_ss_sparse_long))
    # X_seqlens_mat = np.concatenate((X_seqlens_short, X_seqlens_long))

    prot_name=np.concatenate((prot_name_short, prot_name_long))

    ## Class labels
    y=np.array(Info_df.loc[prot_name].RBP_flag)
    y_seq_train = y[:len(prot_name_short)] # Used in training process of DNN, SVM

    print('STEP 1: Preprocessing. Prepare input data for seqSVM.')
    print('Generating sequence features for seqSVM.')
    if union_file == None:
        if no_secondary_structure==False:
            union_file=os.path.join(process_dir, Model_name+'_SVM_SeqFeature_table_comined_all_selected_features_From_WholeDataset.txt')
        else:
            union_file=os.path.join(model_outdir, Model_name+'_SVM_SeqFeatureNoSS_table_comined_all_selected_features_From_WholeDataSet.txt')
        
    if os.path.exists(union_file):
        union_df=pd.read_table(union_file, index_col=0)
    else:
        ## Feature generation and selection for seqSVM.
        ## generate k-mers
        protein_files=list(map(lambda x: os.path.join(seq_dir, x+'.fasta'), prot_name))
        protein_files=[prot_file for prot_file in protein_files if os.path.exists(prot_file)]
        for k in [3,4]:
            Get_kmer_feature_table(k, protein_files, process_dir, RBPs, sequence_type='AA', model_name=Model_name)

        ## generate SS-Kmer
        if no_secondary_structure==False:
            SS_files=list(map(lambda x: os.path.join(seq_dir, x+'.txt') if os.path.exists(os.path.join(seq_dir, x+'.txt')) else os.path.join(seq_dir, x+'.spd3') if os.path.exists(os.path.join(seq_dir, x+'.spd3')) else 'Not_Found', prot_name))
            SS_files=[prot_file for prot_file in SS_files if prot_file!='Not_Found']
            for k in [11,15]:
                Get_kmer_feature_table(k, SS_files, process_dir, RBPs, sequence_type='SS', model_name=Model_name)

        ## generate AAC
        pool1=mp.Pool(processes=None)
        print(protein_files)
        results1 = [pool1.apply_async(get_AAC_features, args=(prot_file,)) for prot_file in protein_files]
        results1= [item.get() for item in results1 if item]
        results1= [x for x in results1 if x]
        results1 = pd.DataFrame(results1).set_index('Protein_Name')
        results1.to_csv(os.path.join(process_dir, Model_name+'_AAC_feature_table.txt'), sep='\t', index=True)

        print('Feature selection for seqSVM.')
        mer3=pd.read_table(os.path.join(process_dir, Model_name+'_AA_3mers_feature_table.txt'),index_col=0).drop('RBP_flag',axis=1)
        mer4=pd.read_table(os.path.join(process_dir, Model_name+'_AA_4mers_feature_table.txt'),index_col=0).drop('RBP_flag',axis=1)
        mer3=mer3.join(Info_df.RBP_flag,how='inner')
        mer4=mer4.join(Info_df.RBP_flag,how='inner')
        if no_secondary_structure==False:
            SS_11mer=pd.read_table(os.path.join(process_dir, Model_name+'_SS_11mers_feature_table.txt'),index_col=0).drop('RBP_flag',axis=1)
            SS_15mer=pd.read_table(os.path.join(process_dir, Model_name+'_SS_15mers_feature_table.txt'),index_col=0).drop('RBP_flag',axis=1)
            SS_11mer=SS_11mer.join(Info_df.RBP_flag,how='inner')
            SS_15mer=SS_15mer.join(Info_df.RBP_flag,how='inner')
        AAC=pd.read_table(os.path.join(process_dir, Model_name+'_AAC_feature_table.txt'),index_col=0)
        AAC=AAC.join(Info_df.RBP_flag,how='inner')

        pool=mp.Pool(processes=None)
        
        ## Feature selection for k-mers
        if no_secondary_structure==False:
            # for mer, name in zip([mer3,mer4,SS_11mer,SS_15mer],['mer3','mer4','SS_11mer','SS_15mer']):
            #     chi2_filtering_followedBy_SVMselection(mer, model_outdir, alpha, Model_name+'_SVM_'+name+'_chi2_alpha'+str(alpha)+'_with_LinearSVC')
            results=[pool.apply(chi2_filtering_followedBy_SVMselection, args=(mer, model_outdir, alpha, Model_name+'_SVM_'+name+'_chi2_alpha'+str(alpha)+'_with_LinearSVC',)) for mer, name in zip([mer3,mer4,SS_11mer,SS_15mer],['mer3','mer4','SS_11mer','SS_15mer'])]
        else:
            results=[pool.apply(chi2_filtering_followedBy_SVMselection, args=(mer, model_outdir, alpha, Model_name+'_SVM_'+name+'_chi2_alpha'+str(alpha)+'_with_LinearSVC',)) for mer, name in zip([mer3,mer4],['mer3','mer4'])]        
        ## Feature selection for AAC
        chi2_filtering_Only(AAC, model_outdir, alpha, Model_name+'_SVM_AAC_chi2_alpha'+str(alpha))

        f=open(os.path.join(model_outdir, Model_name+'_SVM_mer3_chi2_alpha{}_with_LinearSVC_selected_features_From_WholeDataSet.txt'.format(alpha)))
        mer3_selected=f.read().split('\n')
        f.close()
        if mer3_selected==['']:
            mer3_selected=[]
        f=open(os.path.join(model_outdir, Model_name+'_SVM_mer4_chi2_alpha{}_with_LinearSVC_selected_features_From_WholeDataSet.txt'.format(alpha)))
        mer4_selected=f.read().split('\n')
        f.close()
        if mer4_selected==['']:
            mer4_selected=[]
        if no_secondary_structure==False:
            f=open(os.path.join(model_outdir, Model_name+'_SVM_SS_11mer_chi2_alpha{}_with_LinearSVC_selected_features_From_WholeDataSet.txt'.format(alpha)))
            SSmer11_selected=f.read().split('\n')
            f.close()
            if SSmer11_selected==['']:
                SSmer11_selected=[]
            f=open(os.path.join(model_outdir, Model_name+'_SVM_SS_15mer_chi2_alpha{}_with_LinearSVC_selected_features_From_WholeDataSet.txt'.format(alpha)))
            SSmer15_selected=f.read().split('\n')
            f.close()
            if SSmer15_selected==['']:
                SSmer15_selected=[]
        f=open(os.path.join(model_outdir, Model_name+'_SVM_AAC_chi2_alpha{}_selected_features_From_WholeDataSet.txt'.format(alpha)))
        AAC_selected=f.read().split('\n')
        f.close()
        if AAC_selected==['']:
            AAC_selected=[]

        if no_secondary_structure==False:
            # print(mer3[mer3_selected].head(3))
            # print(mer4[mer4_selected].head(3))
            # print(SS_11mer[SSmer11_selected].head(3))
            # print(SS_15mer[SSmer15_selected].head(3))
            # print(AAC[AAC_selected].head(3))
            union_df=mer3[mer3_selected].join(mer4[mer4_selected],how='inner').join(SS_11mer[SSmer11_selected],how='inner').join(SS_15mer[SSmer15_selected],how='inner').join(AAC[AAC_selected],how='inner')
            union_df=union_df.join(Info_df.RBP_flag,how='inner')
            union_df.to_csv(union_file, index=True, sep='\t')
            # union_df.to_csv(os.path.join(process_dir, Model_name+'_SVM_SeqFeature_table_comined_all_selected_features_From_WholeDataset.txt'),index=True,sep='\t')
            with open(os.path.join(model_outdir, Model_name+'_SVM_SeqFeature_all_selected_features_From_WholeDataSet.txt'), 'w') as f:
                f.write('\n'.join(list(union_df.drop('RBP_flag',axis=1).columns)))
        else:
            union_df=mer3[mer3_selected].join(mer4[mer4_selected],how='inner').join(AAC[AAC_selected],how='inner')
            union_df=union_df.join(Info_df.RBP_flag,how='inner')
            union_df.to_csv(union_file,index=True,sep='\t')
            # union_df.to_csv(os.path.join(process_dir, Model_name+'_SVM_SeqFeatureNoSS_table_comined_all_selected_features_From_WholeDataset.txt'),index=True,sep='\t')
            with open(os.path.join(model_outdir, Model_name+'_SVM_SeqFeatureNoSS_all_selected_features_From_WholeDataSet.txt'), 'w') as f:
                f.write('\n'.join(list(union_df.drop('RBP_flag',axis=1).columns)))
        
    ## For SVM
    ### training data
    print('Prepare input data for seqSVM.')
    if 'RBP_flag' in union_df.columns:
        union_df=union_df.drop('RBP_flag',axis=1)
    SVM_train = np.array(union_df.loc[prot_name_short])
    SVM_test = np.array(union_df.loc[prot_name])

    if no_PIA==False:
        print('STEP 1: Preprocessing. Prepare input data for PPI model.')
        print('Generating features for PPI model.')

        PPI_feature_file=os.path.join(process_dir, Model_name+'_PPI_feature_table.txt')
        if not os.path.exists(PPI_feature_file):
            G1=nx.read_edgelist(PPI_edgelist)
            G1.remove_edges_from(nx.selfloop_edges(G1))
            prot_set=set(G1.nodes())
            pool=mp.Pool(processes=None)
            result=[pool.apply_async(get_PPI_features, args=(prot, G1, RBPs,)) for prot in prot_set]
            results=[p.get() for p in result]
            PPI_feature_table=pd.DataFrame(results).set_index('Protein_name')
            PPI_feature_table.to_csv(PPI_feature_file, index=True, sep='\t')
        else:
            PPI_feature_table=pd.read_table(PPI_feature_file,index_col=0)
            prot_set=set(PPI_feature_table.index)

        if no_PPA==False:
            PPI2_feature_file=os.path.join(process_dir, Model_name+'_PIA_feature_table.txt')
            if not os.path.exists(PPI2_feature_file):
                G2=nx.read_edgelist(PPA_edgelist)
                G2.remove_edges_from(nx.selfloop_edges(G2))
                pool=mp.Pool(processes=None)
                result2=[pool.apply_async(get_1stPPI_features, args=(prot, G2, RBPs,)) for prot in prot_set]
                results2=[p.get() for p in result2]
                PPA_feature_table=pd.DataFrame(results2).set_index('Protein_name')
                PIA_feature_table=PPI_feature_table.join(PPA_feature_table, how='left', rsuffix='_PPA')
                PIA_feature_table.to_csv(PPI2_feature_file, index=True, sep='\t')

    if no_PIA==False:
        ## For PPI, MB 
        num_cut=5
        PPI=pd.read_table(PPI_feature_file,index_col=0).drop('RBP_flag',axis=1)[['primary_RBP_ratio','secondary_RBP_ratio','tertiary_RBP_ratio','1st_neighbor_counts']]
        PPI['Reliability']=PPI['1st_neighbor_counts'].apply(lambda x: 1 if x >= num_cut else -1)
        PPI=PPI.drop('1st_neighbor_counts',axis=1)
        PPI_data=np.array(PPI.loc[prot_name])

        if no_PPA==False:
            ## For PPI, MB-STRING (PPI+PPA, PPA: protein-protein association)
            num_cut=5
            PPI2=pd.read_table(PPI2_feature_file,index_col=0).drop('RBP_flag',axis=1)[['primary_RBP_ratio','secondary_RBP_ratio','tertiary_RBP_ratio','primary_RBP_ratio_PPA','1st_neighbor_counts','1st_neighbor_counts_PPA']]
            PPI2['Reliability']=PPI2['1st_neighbor_counts'].apply(lambda x: 1 if x >= num_cut else -1)
            PPI2['Reliability_PPA']=PPI2['1st_neighbor_counts_PPA'].apply(lambda x: 1 if x >= num_cut else -1)
            PPI2=PPI2.drop(['1st_neighbor_counts','1st_neighbor_counts_PPA'],axis=1)
            PPI2_data=np.array(PPI2.loc[prot_name])

    ## For naive ensembling (probabilistic)
    # reference_score_files=reference_score_files.split(',')
    # score_dfs=[]
    # for ref_file in reference_score_files:
    #     score_dfs.append(pd.read_table(ref_file,index_col=0))

    # reference_score_df=pd.concat(score_dfs)

    print('STEP 2: Model training. Pre-train seqDNN.')
    #### Model construction
    if no_pretrain:
        autoEncoder_CNN = autoEncoder_Dense = None
    elif new_pretrain:
        if no_secondary_structure==True:
            dataset_rbp = Protein_Sequence_Input5_noSS(short_files_rbp, class_labels_short_rbp, BioVec_name_dict, maxlen=maxlen)
            X_aa3mer_short_rbp = dataset_rbp.get_aa3mer_mats()
            autoEncoder_CNN, autoEncoder_Dense = autoEncoder_training(process_dir, BioVec_weights_add_null, X_aa3mer_short_rbp, noSS=True, model_name=Model_name)
        else:
            dataset_rbp = Protein_Sequence_Input5(short_files_rbp, class_labels_short_rbp, BioVec_name_dict, maxlen=maxlen)
            X_aa3mer_short_rbp = dataset_rbp.get_aa3mer_mats()
            X_ss_sparse_short_rbp = dataset_rbp.get_ss_sparse_mats2()
            autoEncoder_CNN, autoEncoder_Dense = autoEncoder_training(process_dir, BioVec_weights_add_null, X_aa3mer_short_rbp, X_ss_sparse=X_ss_sparse_short_rbp, noSS=False, model_name=Model_name)

    else:
        autoEncoder_CNN = load_model(autoEncoder_CNN_file)
        autoEncoder_Dense = load_model(autoEncoder_Dense_file)

    if no_secondary_structure:
        DNN_seqOnly_model=SONARp_DNN_SeqOnly_noSS(BioVec_weights_add_null, CNN_trainable=True, maxlen=maxlen, max_seq_len=max_seq_len, dropout=0.3, class_weight={0:1., 1:11.}, batch_size=128, val_fold=None, sliding_step=int((maxlen-2)/2), n_gpus=1, autoEncoder_CNN=autoEncoder_CNN, autoEncoder_Dense=autoEncoder_Dense) #For final model training, use single GPU to avoid model weights inconsistency resulting from multi-GPU model.
    else:
        DNN_seqOnly_model=SONARp_DNN_SeqOnly(BioVec_weights_add_null, CNN_trainable=True, maxlen=maxlen, max_seq_len=max_seq_len, dropout=0.3, class_weight={0:1., 1:11.}, batch_size=128, val_fold=None, sliding_step=int((maxlen-2)/2), n_gpus=1, autoEncoder_CNN=autoEncoder_CNN, autoEncoder_Dense=autoEncoder_Dense) #For final model training, use single GPU to avoid model weights inconsistency resulting from multi-GPU model.
    SVM_seq_model=SVC(kernel='rbf', probability=True, class_weight={1:1.5})
    if no_PIA==False:
        SVM_PPI=SVC(kernel='rbf', probability=True, class_weight={1:1.5})
        if no_PPA == False:
            SVM_PPI2=SVC(kernel='rbf', probability=True, class_weight={1:1.5})

    print('STEP 2: Model training. seqDNN model training.')
    ## Model training
    if no_secondary_structure:
        DNN_seqOnly_model.fit(X_aa3mer_short, y_seq_train)
    else:
        DNN_seqOnly_model.fit(X_aa3mer_short, X_ss_sparse_short, y_seq_train)

    print('STEP 2: Model training. seqSVM model training.')
    SVM_train_over, y_seq_over = oversampling(SVM_train, y_seq_train)
    SVM_seq_model.fit(SVM_train_over, y_seq_over)

    if no_PIA == False:
        print('STEP 2: Model training. PPI/PIA-based model training.')
        PPI_over, y_over = oversampling(PPI_data, y)
        SVM_PPI.fit(PPI_over, y_over)

        if no_PPA == False:
            PPI_over2, y_over2 = oversampling(PPI2_data, y)
            SVM_PPI2.fit(PPI_over2, y_over2)

    print('STEP 2: Model training. Saving model files.')
    ## Save model
    outdir=model_outdir
    DNN_seqOnly_model.save_model2(model_outdir, Model_name+'_seqDNN')
    joblib.dump(SVM_seq_model, os.path.join(model_outdir, Model_name+'_seqSVM_ModelFile.pkl'), protocol=2)
    if no_PIA==False:
        joblib.dump(SVM_PPI, os.path.join(model_outdir, Model_name+'_SVM_PPI_ModelFile.pkl'), protocol=2)
        if no_PPA == False:
            joblib.dump(SVM_PPI2, os.path.join(model_outdir, Model_name+'_SVM_PIA_ModelFile.pkl'), protocol=2)

    ## Prediction
    if no_secondary_structure:
        DNN_pred1_1 = DNN_seqOnly_model.predict_score(X_aa3mer_short, X_seqlens_short)
        DNN_pred1_2 = DNN_seqOnly_model.predict_score(X_aa3mer_long, X_seqlens_long)
    else:
        DNN_pred1_1 = DNN_seqOnly_model.predict_score(X_aa3mer_short, X_ss_sparse_short, X_seqlens_short)
        DNN_pred1_2 = DNN_seqOnly_model.predict_score(X_aa3mer_long, X_ss_sparse_long, X_seqlens_long)
    DNN_pred1 = np.concatenate((DNN_pred1_1, DNN_pred1_2))
    SVM_pred1 = SVM_seq_model.predict_proba(SVM_test)[:,1]
    if no_PIA==False:
        PPI_pred1 = SVM_PPI.predict_proba(PPI_data)[:,1]
        if no_PPA == False:
            PPI2_pred1 = SVM_PPI2.predict_proba(PPI2_data)[:,1]

    score_df = Info_df[['RBP_flag']].loc[prot_name]
    score_df['seqSVM_score'] = SVM_pred1
    score_df['seqDNN_score'] = DNN_pred1
    if no_PIA==False:
        score_df['PPI_score'] = PPI_pred1
        if no_PPA == False:
            score_df['PIA_score'] = PPI2_pred1
    score_df['seqSVM_seqDNN_score'] = -1
    if no_PIA==False:
        score_df['seqDNN_seqSVM_PPI_score'] = -1
        score_df['seqDNNseqSVM_PPI_score'] = -1 # combine intrinsic with extrinsic classifier (seqDNN+seqSVM)+NetMB
        if no_PPA == False:
            score_df['seqDNN_seqSVM_PIA_score'] = -1
            score_df['seqDNNseqSVM_PIA_score'] = -1 # combine intrinsic with extrinsic classifier (seqDNN+seqSVM)+NetMB_STRING

    scores_DNN=list(score_df['seqDNN_score'])
    scores_SVM=list(score_df['seqSVM_score'])
    if no_PIA==False:
        scores_NetMB=list(score_df['PPI_score'])
        if no_PPA == False:
            scores_NetMB_S=list(score_df['PIA_score'])
    true_labels=list(score_df['RBP_flag'])

    for rix, row in score_df.iterrows():
        ## DNN
        TP, FN, TN, FP = get_TP_FN_TN_FP(scores_DNN, true_labels, threshold=row['seqDNN_score'])
        fdr_DNN=get_fdr(TP, FN, TN, FP)
        fpr_DNN=get_fpr(TP, FN, TN, FP)
        ## SVM
        TP, FN, TN, FP = get_TP_FN_TN_FP(scores_SVM, true_labels, threshold=row['seqSVM_score'])
        fdr_SVM=get_fdr(TP, FN, TN, FP)
        fpr_SVM=get_fpr(TP, FN, TN, FP)

        optimistic_prob_intrinsic1 = 1 - fdr_DNN*fdr_SVM

        score_df.loc[rix, 'seqSVM_seqDNN_score'] = optimistic_prob_intrinsic1

    scores_intrinsic=list(score_df['seqSVM_seqDNN_score'])
    if no_PIA==False:
        for rix, row in score_df.iterrows():
             ## DNN
            TP, FN, TN, FP = get_TP_FN_TN_FP(scores_DNN, true_labels, threshold=row['seqDNN_score'])
            fdr_DNN=get_fdr(TP, FN, TN, FP)
            fpr_DNN=get_fpr(TP, FN, TN, FP)
            ## SVM
            TP, FN, TN, FP = get_TP_FN_TN_FP(scores_SVM, true_labels, threshold=row['seqSVM_score'])
            fdr_SVM=get_fdr(TP, FN, TN, FP)
            fpr_SVM=get_fpr(TP, FN, TN, FP)
            ## Network  
            TP, FN, TN, FP = get_TP_FN_TN_FP(scores_NetMB, true_labels, threshold=row['PPI_score'])
            fdr_Net=get_fdr(TP, FN, TN, FP)
            fpr_Net=get_fpr(TP, FN, TN, FP)
            if no_PPA==False:
                TP, FN, TN, FP = get_TP_FN_TN_FP(scores_NetMB_S, true_labels, threshold=row['PIA_score'])
                fdr_Net2=get_fdr(TP, FN, TN, FP)
                fpr_Net2=get_fpr(TP, FN, TN, FP)
            ## Intrinsic
            TP, FN, TN, FP = get_TP_FN_TN_FP(scores_intrinsic, true_labels, threshold=row['seqSVM_seqDNN_score'])
            fdr_in1=get_fdr(TP, FN, TN, FP)

            
            score_df.loc[rix, 'seqDNNseqSVM_PPI_score'] = 1 - fdr_in1*fdr_Net
            if no_PPA==False:
                score_df.loc[rix, 'seqDNNseqSVM_PIA_score'] = 1 - fdr_in1*fdr_Net2
            score_df.loc[rix, 'seqDNN_seqSVM_PPI_score'] = 1 - fdr_DNN*fdr_SVM*fdr_Net
            if no_PPA==False:
                score_df.loc[rix, 'seqDNN_seqSVM_PIA_score'] = 1 - fdr_DNN*fdr_SVM*fdr_Net2


    #uniprotID_mapping=pd.read_table(uniprotID_mapping_file,index_col=0)
    #Final_score_df=uniprotID_mapping.join(score_df,how='right')
    Final_score_df=score_df.join(Info_df['protein_len'], how='left')
    if no_PIA==False:
        if no_PPA==False:
            Final_score_df=Final_score_df.sort_values('seqDNNseqSVM_PIA_score',ascending=False)
        else:
            Final_score_df=Final_score_df.sort_values('seqDNNseqSVM_PPI_score',ascending=False)
    else:
        Final_score_df=Final_score_df.sort_values('seqSVM_seqDNN_score',ascending=False)

    Final_score_df.to_csv(os.path.join(model_outdir, Model_name+'_classification_scores_trainedWithWholeDataset.xls'), sep='\t', index=True)




#if __name__=='__main__':
def call_main():
    usage="""\nHydRa_train -s seq_dir -R filename --PPI_edgelist filename --PPA_edgelist filename""" 
    description="""Run SONAR with your PPI network edge list and RBP annotation list. This program will give you the RCS score table which contains the classification scores for all the proteins appearing in the PPI network. The immediate result (feature table generated in the process) will also be presented."""
    parser= ArgumentParser(usage=usage, description=description)
    #parser.add_option("-h", "--help", action="help")
    parser.add_argument('-M', '--maxlen', dest='maxlen', help='', type=int, default=1500)
    parser.add_argument('-l', '--len_file', dest='len_file', help='', metavar='FILE', default='/home/wjin/projects/SONAR_ChernHan/Data/PPI_data_updated/protein_length_menthaBioPlexSTRING.txt')
    parser.add_argument('-R', '--RBP_list', dest='RBP_list', help='A file contains the names or IDs of the known RBPs. One protein name/ID each line. The names/IDs should be consistent with the names/IDs used in the PPI network and protein sequence fasta files.', metavar='FILE', default=None)
    parser.add_argument('-b', '--BioVec_weights', dest='BioVec_weights', help='', metavar='FILE', default=None)
    parser.add_argument('-s', '--seq_dir', dest='seq_dir', help='The path of the directory for protein sequence and secondary structure files. Protein sequence files should have suffix .fasta, and secondary structure files should have suffix .spd3 or .txt. For flexibility, each .fasta or .spd3/.txt file should contain only one protein.', type=str, default=None)
    parser.add_argument('-T', '--train_list', dest='train_list', help='A file contains the names or IDs of the proteins used to train HydRa. One protein name/ID each line. The names/IDs should be consistent with the names/IDs used in the PPI network and protein sequence fasta files.', metavar='FILE', default=None)
    parser.add_argument('--process_dir', dest='process_dir', help='The path of the directory for storing intermediate files. ', type=str, default='./process_dir')    
    parser.add_argument('--no-secondary-structure', dest='no_secondary_structure', help='Do not use secondary structure information in the model training.', action='store_true')
    parser.add_argument('--no-PIA', dest='no_PIA', help='Do not use protein-protein interaction and functinoal association information in the prediction.', action='store_true')
    parser.add_argument('--no-PPA', dest='no_PPA', help='Do not use protein-protein functinoal association information in the prediction.', action='store_true')
    parser.add_argument('-S', '--seqSVM_ft_file', dest='seqSVM_ft_file', help='', metavar='FILE', default=None)
    parser.add_argument('-g', '--PPI_edgelist', dest='PPI_edgelist', help='PPI network edgelist filepath.', metavar='FILE', default=None)
    parser.add_argument('-G', '--PPA_edgelist', dest='PPA_edgelist', help='Protein-protein association network edgelist filepath.', metavar='FILE', default=None)
    # parser.add_argument('-p', '--PPI_feature_file', dest='PPI_feature_file', help='', metavar='FILE', default='/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/PPI_quality/SONAR_output/menthaBioPlex_feature_table_combined10RBPlist.xls')
    # parser.add_argument('-P', '--PPI2_feature_file', dest='PPI2_feature_file', help='', metavar='FILE', default='/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/PPI_quality/SONAR_output/menthaBioPlex_separatedSTRING_feature_table_combined8RBPlistAddingOOPSXRNAX.xls')
    parser.add_argument('--new-pretrain', dest='new_pretrain', help='Use auto-encoder to pre-train the models from the user input RBPs. If this flag is not activated, the weights of seqDNN will be initialized with user input pre-trained auto-encoder files. If neither pretrain is activated nor the pre-trained auto-encoder files are provided and --no-pretrain is not activated, the auto-encoders pretrained with human RBPs in our HydRa paper will be used to initialize the seqDNN model weights.', action='store_true')
    parser.add_argument('--no-pretrain', dest='no_pretrain', help='No pretrain for seqDNN will be performed and no pretrained weights will be used to initialize seqDNN. seqDNN will be initialized by default initialization algorithm in Keras', action='store_true')
    parser.add_argument('-c', '--autoEncoder_CNN_file', dest='autoEncoder_CNN_file', help='', metavar='FILE', default=None)
    parser.add_argument('-d', '--autoEncoder_Dense_file', dest='autoEncoder_Dense_file', help='', metavar='FILE', default=None)
    parser.add_argument('-n', '--model_name', dest='model_name', help='', type=str, default='HydRa_trained')
    # parser.add_argument('-m', '--uniprotID_mapping_file', dest='uniprotID_mapping_file', help='', metavar='FILE', default='/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/UniprotID_to_GeneName.txt')
    parser.add_argument('-o', '--model_outdir', dest='model_outdir', help='The folder path where model files of the new trained HydRa will be stored.', type=str, default='./models')
    # parser.add_argument('-O', '--score_outdir', dest='score_outdir', help='', type=str, default='/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/SONAR+/')
    parser.add_argument('-a', '--alpha', dest='alpha', help='Threshold for p-value in the feature selection step with Chi-squared test.', type=float, default=0.01)

    args=parser.parse_args()
    if not args.seq_dir:
        parser.error("-s/--seq_dir must be specified.")
    if args.new_pretrain and (args.autoEncoder_CNN_file or args.autoEncoder_Dense_file):
        warnings.warn('Since the --new_pretrain option is activated, the --autoEncoder_CNN_file and --autoEncoder_Dense_file values will be ignored.')
    if args.no_pretrain and (args.new_pretrain or (args.autoEncoder_CNN_file or args.autoEncoder_Dense_file)):
        parser.error("--no-pretrain conflicts with --new_pretrain, --autoEncoder_CNN_file and --autoEncoder_Dense_file.")

    print("Start to train your HydRa model!\nThis program may take several hours, please be patient.")
    main(args)

