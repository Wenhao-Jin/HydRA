#!/usr/bin/env python

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, recall_score, precision_recall_curve, average_precision_score
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
from keras.models import load_model
from keras import regularizers
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.base import clone
import pickle
import re
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing as mp
from random import shuffle
#from propy import PyPro
import joblib
from .models.Sequence_class import Protein_Sequence_Input5, Protein_Sequence_Input5_2, Protein_Sequence_Input5_noSS, Protein_Sequence_Input5_2_noSS
from .models.DNN_seq import SONARp_DNN_SeqOnly, SONARp_DNN_SeqOnly_2, SONARp_DNN_SeqOnly_noSS
from .preprocessing.get_SVM_seq_features import Get_feature_table, Get_feature_table_noSS
from .preprocessing.Feature_generation.PPI_feature_generation import get_PPI_feature_vec, get_1stPPI_feature_vec
from .preprocessing.Feature_generation.Secondary_structure_prediction import predict_2ary_structure_spider2
from argparse import ArgumentParser
import multiprocessing as mp
import networkx as nx

import pkg_resources
import warnings
from pathlib import Path

plt.switch_backend('agg')

FAKE_ZEROS=1e-5


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

	## Prepare the whole dataset for training our final SeqOnly classifier. Also act as test set, so that we can get each protein a prediction score with our trained classifier (even though the classifier is trained with this data itself).
	maxlen=args.maxlen # Used in DNN and SVM_seq's model construction.
	BioVec_weights_file=args.BioVec_weights
	seq_file=args.seq_file
	seq_dir=args.seq_dir
	RBP_list=args.RBP_list
	use_pre_calculated_PPI_feature=args.use_pre_calculated_PPI_feature
	PPI_feature_file=args.PPI_feature_file   # PPI feature table from MB
	PPI2_feature_file=args.PPI2_feature_file  # PPI feature table from MB-STRING
	PPI_edgelist=args.PPI_edgelist
	PPA_edgelist=args.PPA_edgelist
	reference_score_file=args.reference_score_file
	no_secondary_structure=args.no_secondary_structure
	no_PIA=args.no_PIA
	no_PPA=args.no_PPA
	seqDNN_modelfile_stru=args.seqDNN_modelfile_stru
	seqDNN_modelfile_weight=args.seqDNN_modelfile_weight
	seqSVM_modelfile=args.seqSVM_modelfile
	PPI_modelfile=args.PPI_modelfile
	PPI2_modelfile=args.PPI2_modelfile
	score_outdir=args.score_outdir
	Path(score_outdir).mkdir(parents=True, exist_ok=True)
	PPI_1stNB_threshold=args.PPI_1stNB_threshold
	PPI2_1stNB_threshold=args.PPI2_1stNB_threshold
	PPI_1stInteractors_file=args.PPI_1stInteractors_file
	selected_aa3mers_file=args.selected_aa3mers_file
	selected_aa4mers_file=args.selected_aa4mers_file
	selected_SS11mers_file=args.selected_SS11mers_file
	selected_SS15mers_file=args.selected_SS15mers_file
	selected_AAC_file=args.selected_AAC_file
	combined_selected_feature_file=args.combined_selected_feature_file
	HydRa_score_threshold=args.HydRa_score_threshold
	HydRa_score_reference=args.HydRa_score_reference
	FPR=args.FPR
	Model_name=args.model_name
	model_dir=args.model_dir

	if RBP_list == None:
		RBP_list = pkg_resources.resource_string(__name__, 'data/Combined8RBPlist_plusOOPSXRNAXsharedRBPs_uniprotID_20190717.txt').decode("utf-8")
	else:
		with open(RBP_list) as f:
			RBP_list=f.read()
	if BioVec_weights_file == None:
		BioVec_weights_file = pkg_resources.resource_stream(__name__, 'data/protVec_100d_3grams.csv')
	if model_dir == None:
		if no_secondary_structure:
			if seqDNN_modelfile_stru == None:
				seqDNN_modelfile_stru = pkg_resources.resource_string(__name__, 'pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_noSS_WithNoSSmodel_model_structure.json').decode("utf-8")
			else:
				with open(seqDNN_modelfile_stru) as f:
					seqDNN_modelfile_stru=f.read()
			if seqDNN_modelfile_weight == None:
				seqDNN_modelfile_weight = pkg_resources.resource_filename(__name__, 'pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_noSS_WithNoSSmodel_model_weights.h5')
			if seqSVM_modelfile == None:
				seqSVM_modelfile = pkg_resources.resource_stream(__name__, 'pre_trained/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_noSS_WithNoSSmodel_ModelFile.pkl')
		else:
			if seqDNN_modelfile_stru == None:
				seqDNN_modelfile_stru = pkg_resources.resource_string(__name__, 'pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_model_structure.json').decode("utf-8")
			else:
				with open(seqDNN_modelfile_stru) as f:
					seqDNN_modelfile_stru=f.read()
			if seqDNN_modelfile_weight == None:
				seqDNN_modelfile_weight = pkg_resources.resource_filename(__name__, 'pre_trained/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_model_weights.h5')
			if seqSVM_modelfile == None:
				seqSVM_modelfile = pkg_resources.resource_stream(__name__, 'pre_trained/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_ModelFile.pkl')
	else:
		seqDNN_modelfile_stru=os.path.join(model_dir, Model_name+'_seqDNN_model_structure.json')
		with open(seqDNN_modelfile_stru) as f:
				seqDNN_modelfile_stru=f.read()

		seqDNN_modelfile_weight=os.path.join(model_dir, Model_name+'_seqDNN_model_weights.h5')
		seqSVM_modelfile = os.path.join(model_dir, Model_name+'_seqSVM_ModelFile.pkl')
		
	if no_PIA==False:
		if use_pre_calculated_PPI_feature:
			PPI_feature_file = pkg_resources.resource_stream(__name__, 'data/menthaBioPlex_feature_table_combined10RBPlist.xls')
			PPI2_feature_file = pkg_resources.resource_stream(__name__, 'data/menthaBioPlex_separatedSTRING_feature_table_combined8RBPlistAddingOOPSXRNAX.xls')
		else:
			if PPI_edgelist == None:
				PPI_edgelist = pkg_resources.resource_stream(__name__, 'data/mentha20180108_BioPlex2.0_edgelist.txt')
			if no_PPA==False:
				if PPA_edgelist == None:
					PPA_edgelist = pkg_resources.resource_stream(__name__, 'data/STRING_v10.5_uniprot_edgelist_withoutExperimentalData.txt')
		if model_dir == None:
			if PPI_modelfile == None:
				PPI_modelfile = pkg_resources.resource_stream(__name__, 'pre_trained/SVM_PPI_MB_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_ModelFile.pkl')
			if no_PPA==False:
				if PPI2_modelfile == None:
					PPI2_modelfile = pkg_resources.resource_stream(__name__, 'pre_trained/SVM_PPI_MB-S_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_ModelFile.pkl')
		else:
			PPI_modelfile = os.path.join(model_dir, Model_name+'_SVM_PPI_ModelFile.pkl')
			if no_PPA==False:
				PPI2_modelfile = os.path.join(model_dir, Model_name+'_SVM_PIA_ModelFile.pkl')
	if model_dir == None:
		if selected_aa3mers_file == None:
			selected_aa3mers_file = pkg_resources.resource_string(__name__, 'data/SVM_mer3_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt').decode("utf-8")
		else:
			with open(selected_aa3mers_file) as f:
				selected_aa3mers_file = f.read() 
		if selected_aa4mers_file == None:
			selected_aa4mers_file = pkg_resources.resource_string(__name__, 'data/SVM_mer4_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt').decode("utf-8")
		else:
			with open(selected_aa4mers_file) as f:
				selected_aa4mers_file = f.read()
		if no_secondary_structure==False:
			if selected_SS11mers_file == None:
				selected_SS11mers_file = pkg_resources.resource_string(__name__, 'data/SVM_SS_11mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt').decode("utf-8")
			else:
				with open(selected_SS11mers_file) as f:
					selected_SS11mers_file=f.read()
			if selected_SS15mers_file == None:
				selected_SS15mers_file = pkg_resources.resource_string(__name__, 'data/SVM_SS_15mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt').decode("utf-8")
			else:
				with open(selected_SS15mers_file) as f:
					selected_SS15mers_file=f.read()
		if selected_AAC_file == None:
			selected_AAC_file = pkg_resources.resource_string(__name__, 'data/SVM_AAC_chi2_alpha0.01_selected_features_From_WholeDataSet_AACname.txt').decode("utf-8")
		else:
			with open(selected_AAC_file) as f:
				selected_AAC_file=f.read()
		if combined_selected_feature_file == None:
			combined_selected_feature_file = pkg_resources.resource_stream(__name__, 'data/SVM_SeqFeature_table_comined_all_selected_features_From_WholeDataset_12Apr_AACname.txt')
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

		combined_selected_feature_file = os.path.join(model_dir, Model_name+'_SVM_SeqFeature_all_selected_features_From_WholeDataSet.txt')
		with open(combined_selected_feature_file) as f:
			combined_selected_feature=f.read().strip(' \n').split('\n')

	if reference_score_file == None:
		if no_secondary_structure:
			reference_score_file = pkg_resources.resource_stream(__name__, 'data/Classification_cv_scores_AllRuns_SONAR+_final_Ensemble_17_3_7_3_combined10RBPlist_noSS_menthaBioPlex_STRING.tsv')
		else:
			reference_score_file = pkg_resources.resource_stream(__name__, 'data/Classification_cv_scores_AllRuns_SONAR+_final_Ensemble_17_3_7_3_combined10RBPlist_menthaBioPlex_STRING.tsv')
	
	if (HydRa_score_threshold == None) and (HydRa_score_reference == None):
		if no_secondary_structure==False:
			HydRa_score_reference = pkg_resources.resource_stream(__name__, 'data/Classification_scores_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_trainedWithWholeDataset.tsv')
		else:
			HydRa_score_reference = pkg_resources.resource_stream(__name__, 'data/Classification_scores_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_noSS_trainedWithWholeDataset.tsv')


	print('Welcome to HydRa!')
	print('STEP 1: Preprocessing.')
	if not os.path.exists(seq_dir):
		os.mkdir(seq_dir)

	##### Pre-processing
	### Split the sequence file and store it in a new folder
	# with open(seq_file) as f:
	# 	l=[e for e in map(lambda x: x.strip(' \n'), f.read().split('>')) if e !='']

	# for e in l:
	# 	prot=e.split('\n')[0]
	# 	with open(os.path.join(seq_dir, prot+'.fasta') , 'w') as f:
	# 		f.write('>'+e)

	proteins=list(map(lambda x: '.'.join(x.split('.')[:-1]), list(filter(lambda x: x.endswith('.fasta'), os.listdir(seq_dir)))))
	if no_secondary_structure==False:
		proteins2=set((map(lambda x: '.'.join(x.split('.')[:-1]), list(filter(lambda x: x.endswith('.txt') or x.endswith('.spd3'), os.listdir(seq_dir))))))
		proteins=[p for p in proteins if p in proteins2]

	print('Proteins to be predicted: ', proteins)
	
	### Predict secondary structure
	# pool1=mp.Pool(processes=mp.cpu_count())
	# results=[pool1.apply_async(predict_2ary_structure_spider2, args=(os.path.join(seq_dir, prot+'.fasta'), seq_dir)) for prot in proteins0 if os.path.exists(os.path.join(seq_dir, prot+'.fasta')) and not os.path.exists(os.path.join(seq_dir, prot+'.pssm'))]
	# results=[p.get() for p in results]

	# files=list(filter(lambda x: x.endswith('.fasta') and os.path.exists(os.path.join(seq_dir, x.strip('.fasta')+'.spd3')), os.listdir(seq_dir)))
	# proteins=[file.strip('.fasta') for file in files]

	print('STEP 1: Preprocessing. Generating the PPI features')
	### Get PPI features
	if no_PIA==False:
		if not (PPI_feature_file or PPI2_feature_file) and PPI_edgelist:
			G1=nx.read_edgelist(PPI_edgelist)
			G1.remove_edges_from(nx.selfloop_edges(G1))
			RBP_merged_set=set(RBP_list.split('\n'))

			num_cut=PPI_1stNB_threshold
			PPI_features=[]
			if PPI_1stInteractors_file:
				PPI_1stInteractors_df=pd.read_csv(PPI_1stInteractors_file)
				interactors_dic={prot:list(PPI_1stInteractors_df[PPI_1stInteractors_df.Bait==prot]['Preys']) for prot in proteins}
			else:
				interactors_dic={}
			for prot in proteins:
				if prot in interactors_dic:
					PPI_features.append(get_PPI_feature_vec(prot, G1, RBP_merged_set, num_cut, interactors_dic[prot]))
				else:
					PPI_features.append(get_PPI_feature_vec(prot, G1, RBP_merged_set, num_cut))

			PPI_feature_df=pd.DataFrame(PPI_features, index=proteins, columns=['primary_RBP_ratio','secondary_RBP_ratio','tertiary_RBP_ratio','Reliability'])
			PPI_feature_file=os.path.join(score_outdir, Model_name+'_PPI_feature_table.txt')
			PPI_feature_df.to_csv(PPI_feature_file, sep='\t', index=True)

			if PPA_edgelist:
				G2=nx.read_edgelist(PPA_edgelist)
				G2.remove_edges_from(nx.selfloop_edges(G2))
				RBP_merged_set=set(RBP_list.split('\n'))

				num_cut=PPI2_1stNB_threshold
				PAI_features=[]
				for prot in proteins:
				    PAI_features.append(get_1stPPI_feature_vec(prot, G2, RBP_merged_set, num_cut))

				PAI_feature_df=pd.DataFrame(PAI_features, index=proteins, columns=['primary_RBP_ratio','Reliability'])
				PPI2_feature_df=PPI_feature_df.join(PAI_feature_df, how='left', rsuffix='_STRING')

				PPI2_feature_file=os.path.join(score_outdir, Model_name+'_PIA_feature_table.txt')
				PPI2_feature_df.to_csv(PPI2_feature_file, sep='\t', index=True)


		## PPI featrures
		## For PPI, MB 
		if PPI_feature_file:
			PPI=pd.read_table(PPI_feature_file,index_col=0)
			PPI=PPI[['primary_RBP_ratio','secondary_RBP_ratio','tertiary_RBP_ratio','Reliability']]
			proteins=[prot for prot in proteins if prot in PPI.index]
			PPI_data=np.array(PPI.loc[proteins])

		## For PPI, MB-STRING
		if PPI2_feature_file:
			PPI2=pd.read_table(PPI2_feature_file,index_col=0)
			PPI2=PPI2[['primary_RBP_ratio', 'secondary_RBP_ratio', 'tertiary_RBP_ratio',
	       'primary_RBP_ratio_STRING', 'Reliability', 'Reliability_STRING']]
			proteins=[prot for prot in proteins if prot in PPI.index]
			PPI2_data=np.array(PPI2.loc[proteins])
	else:
		PPI_feature_file=PPI2_feature_file=None

	print('STEP 1: Preprocessing. Preparing the sequence-based features.')
	## DNN features
	BioVec_weights=pd.read_table(BioVec_weights_file, sep='\t', header=None, index_col=0)
	BioVec_weights_add_null=np.append(np.zeros((1,100)), BioVec_weights.values, axis=0) #append a [0,0,...,0] array at the top of the matrix, which used for padding 0s.
	BioVec_weights_add_null=BioVec_weights_add_null*10
	BioVec_name_dict={}
	for i in range(1, len(BioVec_weights)+1):
		BioVec_name_dict.update({BioVec_weights.index[i-1]:i})
	
	seq_dic={}
	if no_secondary_structure==False:
		ss_seq_dic={}
	for prot in proteins:
		with open(os.path.join(seq_dir,prot+'.fasta')) as f:
			seq=''.join(f.read().strip('* \n').split('\n')[1:])
			seq=seq.replace('*','')
			seq=seq.replace('X','')

		seq_dic[prot]=seq
		if no_secondary_structure==False:
			if os.path.exists(os.path.join(seq_dir,prot+'.spd3')):
				tmp=pd.read_table(os.path.join(seq_dir,prot+'.spd3'))
				tmp=tmp[tmp['AA']!='*']
				tmp=tmp[tmp['AA']!='X']
				ss=''.join(list(tmp['SS'])).strip(' ')
			else:
				with open(os.path.join(seq_dir,prot+'.txt')) as f:
					ss=''.join(f.read().strip('* \n').split('\n')[1:])

			ss_seq_dic[prot]=ss
		
		

	seq_names=seq_dic.keys()
	class_labels=np.ones(len(seq_names))
	prot_seqs=[seq_dic[prot] for prot in seq_names]
	if no_secondary_structure==False:
		ss_seqs=[ss_seq_dic[prot] for prot in seq_names]

	if no_secondary_structure:
		RBP=Protein_Sequence_Input5_2_noSS(seq_names, prot_seqs, class_labels, BioVec_name_dict, maxlen=maxlen)
		RBP_aa3mer=RBP.get_aa3mer_mats()
		RBP_seqlens = RBP.get_seqlens()
		max_seq_len = RBP.get_maxlen()
	else:
		RBP=Protein_Sequence_Input5_2(seq_names, prot_seqs, ss_seqs, class_labels, BioVec_name_dict, maxlen=maxlen)
		RBP_aa3mer=RBP.get_aa3mer_mats()#[:900]
		RBP_ss_sparse_mat = RBP.get_ss_sparse_mats2()#[:900,:]
		RBP_seqlens = RBP.get_seqlens()
		max_seq_len = RBP.get_maxlen()

	## SVM features
	selected_aa3mers=selected_aa3mers_file.split('\n')
	selected_aa4mers=selected_aa4mers_file.split('\n')
	if no_secondary_structure==False:
		selected_SS11mers=selected_SS11mers_file.split('\n')
		selected_SS15mers=selected_SS15mers_file.split('\n')

	selected_AAC=selected_AAC_file.split('\n')
	selected_AAC=[x for x in selected_AAC if x]
      
	if no_secondary_structure:
		combined_selected_feature =list(filter(lambda x: len(x)<11, combined_selected_feature))
		seqSVM_ft=np.array([Get_feature_table_noSS(seq_dic[prot], selected_aa3mers, selected_aa4mers, selected_AAC, combined_selected_feature) for prot in proteins])
	else:
		seqSVM_ft=np.array([Get_feature_table(seq_dic[prot], ss_seq_dic[prot], selected_aa3mers, selected_aa4mers, selected_SS11mers, selected_SS15mers, selected_AAC, combined_selected_feature) for prot in proteins])

	pd.DataFrame(seqSVM_ft, columns=combined_selected_feature, index=proteins).to_csv(os.path.join(score_outdir, Model_name+'_seqSVM_feature_table.txt'), sep='\t', index=True)
	
	print('STEP 2: Loading HydRa models.')
	## Load models
	if no_secondary_structure:
		model_DNN=SONARp_DNN_SeqOnly_noSS(BioVec_weights_add_null=BioVec_weights_add_null, CNN_trainable=False, maxlen=maxlen, max_seq_len=max_seq_len, dropout=0.3, class_weight={0:1., 1:11.}, batch_size=128, val_fold=None, sliding_step=int((maxlen-2)/2), n_gpus=1)
	else:
		model_DNN=SONARp_DNN_SeqOnly(BioVec_weights_add_null=BioVec_weights_add_null, CNN_trainable=False, maxlen=maxlen, max_seq_len=max_seq_len, dropout=0.3, class_weight={0:1., 1:11.}, batch_size=128, val_fold=None, sliding_step=int((maxlen-2)/2), n_gpus=1)

	model_DNN.load_model2(seqDNN_modelfile_stru, seqDNN_modelfile_weight)

	model_SVM=joblib.load(seqSVM_modelfile)
	model_SVM.kernel='rbf' # To fix, sklearn conflict issue between python3 and python2 model string.

	if PPI_feature_file:
		model_PPI=joblib.load(PPI_modelfile)
		model_PPI.kernel='rbf'

	if PPI2_feature_file:
		model_PPI2=joblib.load(PPI2_modelfile)
		model_PPI2.kernel='rbf'

	print('STEP 3: Making predictions.')
	if no_secondary_structure:
		seqDNN_score=model_DNN.predict_score(RBP_aa3mer, RBP_seqlens)
	else:
		seqDNN_score=model_DNN.predict_score(RBP_aa3mer, RBP_ss_sparse_mat, RBP_seqlens)
	seqSVM_score=model_SVM.predict_proba(seqSVM_ft)[:,1]
	if PPI_feature_file:
		PPI_score = model_PPI.predict_proba(PPI_data)[:,1]
	if PPI2_feature_file:
		PPI2_score = model_PPI2.predict_proba(PPI2_data)[:,1]


	#### Load reference score table (use for FDR calculation)
	reference_score_df=pd.read_table(reference_score_file, index_col=0)

	reference_score_df['seqSVM_seqDNN_score']=-1
	scores_DNN=list(reference_score_df['seqDNN_score'])
	scores_SVM=list(reference_score_df['seqSVM_score'])
	true_labels=list(reference_score_df['RBP_flag'])
	for rix, row in reference_score_df.iterrows():
		## DNN
		TP, FN, TN, FP = get_TP_FN_TN_FP(scores_DNN, true_labels, threshold=row['seqDNN_score'])
		fdr_DNN=get_fdr(TP, FN, TN, FP)
		## SVM
		TP, FN, TN, FP = get_TP_FN_TN_FP(scores_SVM, true_labels, threshold=row['seqSVM_score'])
		fdr_SVM=get_fdr(TP, FN, TN, FP)

		reference_intrinsic = 1 - fdr_DNN*fdr_SVM
		reference_score_df.loc[rix, 'seqSVM_seqDNN_score'] = reference_intrinsic

	scores_intrinsic=list(reference_score_df['seqSVM_seqDNN_score'])

	score_df=pd.DataFrame(index=proteins)
	score_df['seqSVM_score'] = seqSVM_score
	score_df['seqDNN_score'] = seqDNN_score
	
	score_df['seqSVM_seqDNN_score'] = -1
	if PPI_feature_file:
		score_df['PPI_score'] = PPI_score
		score_df['seqDNN_seqSVM_PPI_score'] = -1
		score_df['seqDNNseqSVM_PPI_score'] = -1
		scores_NetMB=list(reference_score_df['PPI_score'])
	if PPI2_feature_file:
		score_df['PIA_score'] = PPI2_score
		score_df['seqDNN_seqSVM_PIA_score'] = -1
		score_df['seqDNNseqSVM_PIA_score'] = -1
		scores_NetMB_S=list(reference_score_df['PIA_score'])

	scores_DNN=list(reference_score_df['seqDNN_score'])
	scores_SVM=list(reference_score_df['seqSVM_score'])
	scores_NetMB=list(reference_score_df['PPI_score'])
	scores_NetMB_S=list(reference_score_df['PIA_score'])
	true_labels=list(reference_score_df['RBP_flag'])
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
		if PPI_feature_file:
			TP, FN, TN, FP = get_TP_FN_TN_FP(scores_NetMB, true_labels, threshold=row['PPI_score'])
			fdr_Net=get_fdr(TP, FN, TN, FP)
			fpr_Net=get_fpr(TP, FN, TN, FP)
		if PPI2_feature_file:
			TP, FN, TN, FP = get_TP_FN_TN_FP(scores_NetMB_S, true_labels, threshold=row['PIA_score'])
			fdr_Net2=get_fdr(TP, FN, TN, FP)
			fpr_Net2=get_fpr(TP, FN, TN, FP)
		## Intrinsic
		optimistic_prob_intrinsic1 = 1 - fdr_DNN*fdr_SVM
		TP, FN, TN, FP = get_TP_FN_TN_FP(scores_intrinsic, true_labels, threshold=optimistic_prob_intrinsic1)
		fdr_in1=get_fdr(TP, FN, TN, FP)


		optimistic_prob_intrinsic1 = 1 - fdr_DNN*fdr_SVM
		score_df.loc[rix, 'seqSVM_seqDNN_score'] = optimistic_prob_intrinsic1
		if PPI_feature_file:
			optimistic_prob2 = 1 - fdr_DNN*fdr_SVM*fdr_Net
			optimistic_prob5 = 1 - fdr_in1*fdr_Net
			score_df.loc[rix, 'seqDNN_seqSVM_PPI_score'] = optimistic_prob2
			score_df.loc[rix, 'seqDNNseqSVM_PPI_score'] = optimistic_prob5
		if PPI2_feature_file:
			optimistic_prob4 = 1 - fdr_DNN*fdr_SVM*fdr_Net2
			optimistic_prob6 = 1 - fdr_in1*fdr_Net2
			score_df.loc[rix, 'seqDNN_seqSVM_PIA_score'] = optimistic_prob4
			score_df.loc[rix, 'seqDNNseqSVM_PIA_score'] = optimistic_prob6

	if PPI2_feature_file:
		score_df=score_df.sort_values('seqDNNseqSVM_PIA_score',ascending=False)
	elif PPI_feature_file:
		score_df=score_df.sort_values('seqDNNseqSVM_PPI_score',ascending=False)
	else:
		score_df=score_df.sort_values('seqSVM_seqDNN_score',ascending=False)


	## Give final prediction based on HydRa score threshold
	if HydRa_score_threshold==None:
		ref_df=pd.read_table(HydRa_score_reference,index_col=0)
		ref_df0=ref_df[ref_df.RBP_flag==0]
		if PPI2_feature_file:
			col='seqDNNseqSVM_PIA_score'
		elif PPI_feature_file:
			col='seqDNNseqSVM_PPI_score'
		else:
			col='seqSVM_seqDNN_score'
		ref_df0=ref_df0.sort_values(col, ascending=False)
		print(col)
		HydRa_score_threshold=ref_df0.iloc[round(len(ref_df0)*FPR)][col]
		print("HydRa_score_threshold is set as {}.".format(HydRa_score_threshold))

	# score_df=score_df.drop('RBP_flag',axis=1)
	score_df['HydRa_RBPs']=score_df[col].apply(lambda x: True if x >= HydRa_score_threshold else False)
	score_df.to_csv(os.path.join(score_outdir, Model_name+'_HydRa_predictions.csv'), sep=',', index=True)
	print('STEP 4: Prediction Done. If the results look good, pleasing sending beer to ...')





def call_main():
	usage="""\n./HydRa_predict.py -p PPI_feature_file -P PPI2_feature_filename -f seq_file""" 
	description="""Use trained HydRa to predict the RNA-binding capacity of given proteins. ps: If at least one of the -p/--PPI_feature_file or -P/--PPI2_feature_file is provided, then PPI_edgelist and PPA_edgelist will be ignored."""
	parser= ArgumentParser(usage=usage, description=description)
	#parser.add_option("-h", "--help", action="help")
	parser.add_argument('-M', '--maxlen', dest='maxlen', help='', type=int, default=1500)
	parser.add_argument('-b', '--BioVec_weights', dest='BioVec_weights', help='', metavar='FILE', default=None)
	parser.add_argument('-f', '--seq_file', dest='seq_file', help='fasta file of the given proteins.', type=str, default='')
	parser.add_argument('-s', '--seq_dir', dest='seq_dir', help='directory for processed sequence and secondary structure files.', type=str, default='./processed')
	parser.add_argument('--use_pre_calculated_PPI_feature', dest='use_pre_calculated_PPI_feature', action='store_true')
	parser.add_argument('--no-secondary-structure', dest='no_secondary_structure', help='Do not use secondary structure information in the prediction.', action='store_true')
	parser.add_argument('--no-PIA', dest='no_PIA', help='Do not use protein-protein interaction and functinoal association information in the prediction.', action='store_true')
	parser.add_argument('--no-PPA', dest='no_PPA', help='Do not use protein-protein functinoal association information in the prediction.', action='store_true')
	parser.add_argument('-p', '--PPI_feature_file', dest='PPI_feature_file', help='Experimental PPI', metavar='FILE', default=None)
	parser.add_argument('-P', '--PPI2_feature_file', dest='PPI2_feature_file', help='ExperimentalPPI + predictedPPI', metavar='FILE', default=None)
	parser.add_argument('--model_dir', dest='model_dir', help='The filepath of the folder stores all the model files that required to run HydRa.', metavar='FILE', default=None)
	parser.add_argument('-D', '--seqDNN_modelfile_stru', dest='seqDNN_modelfile_stru', help='', metavar='FILE', default=None)
	parser.add_argument('-d', '--seqDNN_modelfile_weight', dest='seqDNN_modelfile_weight', help='', metavar='FILE', default=None)
	parser.add_argument('-S', '--seqSVM_modelfile', dest='seqSVM_modelfile', help='', metavar='FILE', default=None)
	parser.add_argument('-m', '--PPI_modelfile', dest='PPI_modelfile', help='Model file for PPI model with MenthaBioPlex network', metavar='FILE', default=None)
	parser.add_argument('-H', '--PPI2_modelfile', dest='PPI2_modelfile', help='Model file for PPI model with MenthaBioPlex network and STRING network.', metavar='FILE', default=None)
	parser.add_argument('-t', '--PPI_1stNB_threshold', dest='PPI_1stNB_threshold', help='Threshold of 1st-level PPI neighbors for proteins with reliable PPI infomation, should be matched with threshold the training session.', type=int, default=5)
	parser.add_argument('-T', '--PPI2_1stNB_threshold', dest='PPI2_1stNB_threshold', help='Threshold of 1st-level predicted PPI (protein association network) neighbors for proteins with reliable PPI infomation, should be matched with threshold the training session.', type=int, default=5)
	parser.add_argument('-r', '--reference_score_file', dest='reference_score_file', help='The file path of the reference score file.', metavar='STRING', default=None)
	parser.add_argument('-O', '--outdir', dest='score_outdir', help='The path of the folder that will store the prediction results.', type=str, default='./prediction_out')
	parser.add_argument('-R', '--RBP_list', dest='RBP_list', help='A file contains the names or IDs of the known RBPs. One protein name/ID each line. The names/IDs should be consistent with the names/IDs used in the PPI network.', metavar='FILE', default=None)
	parser.add_argument('-g', '--PPI_edgelist', dest='PPI_edgelist', help='PPI network edgelist filepath.', metavar='FILE', default=None)
	parser.add_argument('-G', '--PPA_edgelist', dest='PPA_edgelist', help='Protein-protein association network edgelist filepath.', metavar='FILE', default=None)
	parser.add_argument('-I', '--PPI_1stInteractors_file', dest='PPI_1stInteractors_file', help='edgelist showing the 1st-level interactors of given proteins.', metavar='FILE', default=None)
	parser.add_argument('--selected_aa3mers_file', dest='selected_aa3mers_file', help='', metavar='FILE', default=None)
	parser.add_argument('--selected_aa4mers_file', dest='selected_aa4mers_file', help='', metavar='FILE', default=None)
	parser.add_argument('--selected_SS11mers_file', dest='selected_SS11mers_file', help='', metavar='FILE', default=None)
	parser.add_argument('--selected_SS15mers_file', dest='selected_SS15mers_file', help='', metavar='FILE', default=None)
	parser.add_argument('--selected_AAC_file', dest='selected_AAC_file', help='', metavar='FILE', default=None)
	parser.add_argument('--combined_selected_feature_file', dest='combined_selected_feature_file', help='', metavar='FILE', default=None)
	parser.add_argument('-n', '--model_name', dest='model_name', help='A customized name of this prediction made by the user. This prediction_name will be the prefix of the filenames for the prediction output.', type=str, default='RBP_prediction')
	parser.add_argument('--HydRa_score_reference', dest='HydRa_score_reference', help='The score table file that will be used to define the HydRa score threshold for final RBP prediction with the level of false positive rate defined in --FPR option. This option will be ignored if the --HydRa_score_threshold value is provided.', type=str, default=None)
	parser.add_argument('--FPR', dest='FPR', help='The Level of false positive rate that will be used to control the HydRa prediction results.', type=str, default=0.1)
	parser.add_argument('-c', '--HydRa_score_threshold', dest='HydRa_score_threshold', help='Specify the HydRa score cutoff for final RBP prediction. The default value is inferred from our pretrained HydRa model in our paper with false positive rate set as 10%.', type=str, default=None)

	args=parser.parse_args()

	if not (args.PPI_feature_file or args.PPI2_feature_file or args.PPI_edgelist or args.PPA_edgelist):
		print('No PPI information is provided. HydRa will run with only sequence-based information.\n')
	if not args.seq_dir:
		parser.error("-s/--seq_dir must be specified.")    

	if args.use_pre_calculated_PPI_feature and (args.PPI_feature_file or args.PPI2_feature_file):
		warnings.warn('The --use_pre_calculated_PPI_feature option is activated such that the --PPI_feature_file and --PPI2_feature_file values will be ignored.')
	
	if args.model_dir:
		if (seqDNN_modelfile_stru!=None) or (seqDNN_modelfile_weight!=None) or (seqSVM_modelfile!=None) or (PPI_modelfile!=None) or (PPI2_modelfile!=None) or (selected_aa3mers_file!=None) or (selected_aa4mers_file!=None) or (selected_SS11mers_file!=None) or (selected_SS15mers_file!=None) or (selected_AAC_file!=None) or (combined_selected_feature_file!=None):
			warnings.warn('The --model_dir option has been specified. The model files and selected features list will be extracted from this folder. The other input files for models and selected features will be ignored.')
	print("HydRa has started to make predictions on your proteins!\n")
	main(args)

if __name__=='__main__':
	call_main()
