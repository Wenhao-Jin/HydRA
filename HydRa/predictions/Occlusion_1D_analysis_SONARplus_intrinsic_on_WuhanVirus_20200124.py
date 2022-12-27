#!/usr/bin/env python
## v2: simplified the computation of SVM k-mer/SSkmer count updates.


import pandas as pd
import pandas as pd
import numpy as np
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
from sklearn.externals import joblib

plt.switch_backend('agg')

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

aa_code={'H':1, 'K':2, 'D':3, 'E':4,
                 'S':5, 'T':6, 'N':7, 'Q':8, 'C':9,
                 'U':10, 'G':11, 'P':12, 'A':13, 'V':14,
                 'I':15, 'L':16, 'M':17, 'F':18, 'Y':19,
                 'W':20, 'R':21, 'blank':0}
aa_code_reverse={v:k for k, v in aa_code.items()}
BioVec_weights=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data/protVec_100d_3grams.csv', sep='\t', header=None, index_col=0)
BioVec_weights_add_null=np.append(np.zeros((1,100)), BioVec_weights.values, axis=0) #append a [0,0,...,0] array at the top of the matrix, which used for padding 0s.
BioVec_weights_add_null=BioVec_weights_add_null*10
BioVec_name_dict={}

for i in range(1, len(BioVec_weights)+1):
    BioVec_name_dict.update({BioVec_weights.index[i-1]:i})
    
BioVec_name_reverse_dict={v:k for k, v in BioVec_name_dict.items()}

#PPI_FSweight_df=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data/PPI_FSweight_feature_table_BioPlex_combined8RBPlist_2.xls', index_col=0)

class Protein_Sequence_Input5:
    """
    Different from version Protein_Sequence_Input4: Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding amino acid (from aa3mer)
    """
    def __init__(self, files, class_labels, BioVec_name_dict, max_seqlen=1000):
        """
        files: a list of sequence filenames including the absolute path, best in numpy.array format.
        class_labels: a list recording the RBP identity (True or False) of each protein mentioned in "files", best in numpy.array format.
        PPI_feature_vectors: a list of PPI feature vectors for each protein mentioned in "files", best in numpy.array format.
        BioVec_name_dict: dictionary, a dictory telling the mapping between the aa triplet (e.g, HKE) and their index in this analysis.
        max_seqlen: int, the maximum length of amino acids that can be retained.
        """
        if len(files)!=len(class_labels) :
            raise ValueError('The length of files list and class_labels list should be same.')
            
        self.aa_code={'H':1, 'K':2, 'D':3, 'E':4,
                 'S':5, 'T':6, 'N':7, 'Q':8, 'C':9,
                 'U':10, 'G':11, 'P':12, 'A':13, 'V':14,
                 'I':15, 'L':16, 'M':17, 'F':18, 'Y':19,
                 'W':20, 'R':21}
        self.ss_code={'E':1, 'H':2, 'C':3}
        self.ss_sparse_code={'E':[1,0,0], 'H':[0,1,0], 'C':[0,0,1]}
        self.ss_mer_code={'EEE':1, 'HHH':2, 'CCC':3,
                          'EEH':4, 'EEC':5,'EHE':6,'EHH':7,'EHC':8,'ECE':9,'ECH':10,'ECC':11,
                          'HEE':12,'HEH':13,'HEC':14,'HHE':15,'HHC':16,'HCE':17,'HCH':18,'HCC':19,
                          'CEE':20,'CEH':21,'CEC':22,'CHE':23,'CHH':24,'CHC':25,'CCE':26,'CCH':27} #use number 28 to represent unknown 3-mers
        self.max_seq_len=max_seqlen
        self.BioVec_name_dict=BioVec_name_dict
        self.BioVec_name_keys=self.BioVec_name_dict.keys()
        self.prot_names=[]
        self.seqs=[]
        self.seqlen=[]
        #self.PseAAC_mat=[]
        self.seq_mats=[]
        self.ss_mats=[]
        self.ss_sparse_mats=[]
        self.ss_sparse_mats2=[] #Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding aa3mer
        self.aa_ss_mixed_mats=[]
        self.prot_3mer_mats=[]
        self.ss_3mer_mats=[]
        #self.seq_train_mats=[]
        #self.seq_val_mats=[]
        self.labels=[]
        self.train_labels=[]
        self.val_labels=[]
        self.usable_files=[]
        self.batch_id=0
        #max_seq=0
        for seq_file, class_label in zip(files, class_labels):
            try:
                seq, ss_seq, seq_len = self.get_sequence(seq_file)
                self.seqs.append((seq, ss_seq))
                self.seqlen.append(seq_len)
                self.labels.append(class_label)
                self.usable_files.append(seq_file)
                if seq_len>self.max_seq_len:
                    self.max_seq_len=seq_len
                    #max_seq=seq_file

            except ValueError:
                print(seq_file+" is ignored, because of the length conflict in seq_file and ss_seq_file.")
        
        #print max_seq
        
        if len(self.seqs)!=len(self.labels):
            raise ValueError('The length of generated seq vector and class_labels list should be same.')
            
        print("The maximum length of the sequences is : "+str(self.max_seq_len))
        
        for seq, ss_seq in self.seqs:
            seq_mat, ss_mat, aa_ss_mixed_mat, prot_3mer_mat, ss_3mer_mat, ss_sparse_mat, ss_sparse_mat2 = self.encode_protein(seq, ss_seq)
            self.seq_mats.append(seq_mat)
            self.ss_mats.append(ss_mat)
            self.aa_ss_mixed_mats.append(aa_ss_mixed_mat)
            self.prot_3mer_mats.append(prot_3mer_mat)
            self.ss_3mer_mats.append(ss_3mer_mat)
            self.ss_sparse_mats.append(ss_sparse_mat)
            self.ss_sparse_mats2.append(ss_sparse_mat2)
            
        #TODO: Split into training, validation and test dataset
        self.seq_mats=np.array(self.seq_mats)
        self.ss_mats=np.array(self.ss_mats)
        self.ss_sparse_mats=np.array(self.ss_sparse_mats)
        self.ss_sparse_mats2=np.array(self.ss_sparse_mats2)
        self.aa_ss_mixed_mats=np.array(self.aa_ss_mixed_mats)
        self.prot_3mer_mats=np.array(self.prot_3mer_mats)
        self.ss_3mer_mats=np.array(self.ss_3mer_mats)
        #self.PseAAC_mat=np.array(self.PseAAC_mat)
        self.seqlen=np.array(self.seqlen)
        self.labels=np.array(self.labels)
        self.usable_files=np.array(self.usable_files)

        
    def get_sequence(self, seq_file):
        #print seq_file
        f=open(seq_file, 'r')
        s=f.read()
        try:
            s=s.split('\n')[1]
            s=s.replace('*','')
            s=s.replace('X','')
        except:
            print("invalid sequence file:"+seq_file)
            f.close()
            return None, None
            
        f.close()
        seq_file2='.'.join(seq_file.split('.')[:-1])+'.spd3'
        tmp=pd.read_table(seq_file2)
        try:
            tmp=tmp[tmp['AA']!='*']
            tmp=tmp[tmp['AA']!='X']
            ss=''.join(list(tmp['SS']))
            if len(s)!=len(ss):
                #print len(s), len(ss)
                raise ValueError('The length of sequence and SS sequence is different. Files: '+seq_file+', '+seq_file2+'.')
        except:
            print('Problem with {}'.format(seq_file2))
            raise ValueError('Problem with {}'.format(seq_file2))

        return s, ss, len(s)
    
    def encode_amino_acid(self, aa): #convert the AA to a integer, rather than a binary vector.
        aa=aa.upper()
        v=self.aa_code[aa]
        return v

    def encode_secondary_structure(self, ss):
        ss=ss.upper()
        v=self.ss_code[ss]
        return v
    
    def encode_secondary_structure_sparse(self, ss):
        ss=ss.upper()
        v=self.ss_sparse_code[ss]
        return v
    
    def encode_aa_3mer(self, mer):
        mer=mer.upper()
        if mer in self.BioVec_name_keys:
            v=self.BioVec_name_dict[mer]
        else:
            v=9048 # use the index of '<unk>' from the BioVec weights matrix.
        return v
    
    def encode_ss_3mer(self, mer):
        mer=mer.upper()
        if mer in self.ss_mer_code:
            v=self.ss_mer_code[mer]
        else:
            v=28
        return v
    
    def encode_protein(self, prot_seq, ss_seq):
        """
        return: protein_sequence_using_digits, secondary_structure_sequence_using_digits, mixed_aa_ss_using_digits, protein_3mer_sequence_using_digits
        """
        length=len(prot_seq)
        prot_seq_l=[self.encode_amino_acid(aa) for aa in prot_seq]
        prot_seq_l+=[0]*(self.max_seq_len-length)
        ss_seq_l=[self.encode_secondary_structure(ss) for ss in ss_seq]
        ss_seq_l+=[0]*(self.max_seq_len-length)
        ss_seq_sparse_l=[self.encode_secondary_structure_sparse(ss) for ss in ss_seq]
        ss_seq_sparse2_l=ss_seq_sparse_l[1:-1]
        ss_seq_sparse_l+=[[0,0,0]]*(self.max_seq_len-length)
        ss_seq_sparse2_l+=[[0,0,0]]*(self.max_seq_len-length)
        
        prot_3mer_seq_l=[]
        for i in range(len(prot_seq)-2):
            prot_3mer_seq_l.append(self.encode_aa_3mer(prot_seq[i:i+3]))
        
        #prot_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        prot_3mer_seq_l+=[0]*(self.max_seq_len-(length))
        
        ss_3mer_seq_l=[]
        for i in range(len(ss_seq)-2):
            ss_3mer_seq_l.append(self.encode_ss_3mer(ss_seq[i:i+3]))
        
        #ss_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        ss_3mer_seq_l+=[0]*(self.max_seq_len-(length))


        #return csr_matrix(np.array(seq_l))
        return np.array(prot_seq_l), np.array(ss_seq_l), np.array(zip(prot_seq_l, np.array(ss_seq_l)+21)).flatten(), np.array(prot_3mer_seq_l), np.array(ss_3mer_seq_l), np.array(ss_seq_sparse_l), np.array(ss_seq_sparse2_l)
        
    
    def get_aa3mer_mats(self):
        return self.prot_3mer_mats

    def get_ss3mer_mats(self):
        return self.ss_3mer_mats
    
    def get_ss_sparse_mats(self):
        return self.ss_sparse_mats

    def get_ss_sparse_mats2(self): #Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding aa3mer
        return self.ss_sparse_mats2
    
    def get_class_labels(self):
        return self.labels

    def get_seqlens(self):
        return self.seqlen

    def get_usable_files(self):
        return self.usable_files

    def get_maxlen(self):
        return self.max_seq_len


class SONARp_DNN_SeqOnly:
    def __init__(self, BioVec_weights_add_null, max_seq_len, CNN_trainable=True, class_weight={0:1., 1:9.}, dropout=0.3, maxlen=1500, batch_size=50, val_fold=10, sliding_step=100, optimizer='Adam',n_gpus=1,CPU='/device:CPU:0', denses=[10,5]):
        """
        BioVec_weights_add_null: numpy.array, embedding weights matrix for aa triplets.
        CNN_trainable: Boolean, whether allow weights updated in CNN section during model training.
        class_weights: Dictionary, weights for different class. Used in loss function of DNN training.
        dropout: float/double, the dropout used in DNN model.
        maxlen: int, the maximum length of the sequence that will not activate sliding window mode. PS: This model uses aa-3mer as input features, so the effective length of the maximum length is maxlen-2.
        batch_size: int, the batch size in DNN model training.
        val_fold: int, the k for k-fold split of the training set to training subset and validation set. if None, there is not validation set.
        sliding_step: int, the step length in sliding window mode, which is used for long proteins sequence (longer than maxlen).
        """
        self.class_weight=class_weight
        self.maxlen=int(maxlen)
        self.max_seq_len=max_seq_len
        if self.max_seq_len:
            self.max_seq_len = int(self.max_seq_len)
        self.dropout=dropout
        self.val_fold=val_fold
        self.CNN_trainable=CNN_trainable
        self.batch_size=batch_size
        self.class_weight=class_weight
        self.optimizer=optimizer
        self.BioVec_weights_add_null=BioVec_weights_add_null
        self.sliding_step=sliding_step
        self.denses=denses
        self.n_gpus=n_gpus
        self.CPU=CPU
        self.model=self.get_model()
        ## set up models for long input.
        self.model_long_input = {}
        if self.max_seq_len and self.max_seq_len>self.maxlen:
            input_length=int(self.maxlen+self.maxlen/2)
            while input_length <= (self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+(self.maxlen/2):
                self.model_long_input[min(input_length, self.max_seq_len)]=self.get_model_long_prot(min(input_length, self.max_seq_len))
                input_length+=int(self.maxlen/2)

    def fit(self, X_aa3mer_train, X_ss_sparse_train, y_train):

        if self.val_fold:
            skf=StratifiedKFold(n_splits=self.val_fold)
            train, val = list(skf.split(X_aa3mer_train, y_train))[0]
            X_aa3mer_val=X_aa3mer_train[val]
            X_aa3mer_val=X_aa3mer_val[:,:self.maxlen-2]
            X_ss_sparse_val=X_ss_sparse_train[val]
            X_ss_sparse_val=X_ss_sparse_val[:,:self.maxlen-2]
            y_val=y_train[val]
            X_aa3mer_train=X_aa3mer_train[train]
            X_aa3mer_train=X_aa3mer_train[:,:self.maxlen-2]
            X_ss_sparse_train=X_ss_sparse_train[train]
            X_ss_sparse_train=X_ss_sparse_train[:,:self.maxlen-2]
            y_train=y_train[train]
            #print X_aa3mer_train.shape, X_ss_sparse_train.shape, y_train.shape
            
            #early_stopping=EarlyStopping(monitor='val_matthews_correlation', patience=8)
            history=self.model.fit([X_aa3mer_train, X_ss_sparse_train], np_utils.to_categorical(y_train),
              batch_size=self.batch_size, \
              nb_epoch=200, \
              validation_data=([X_aa3mer_val, X_ss_sparse_val], np_utils.to_categorical(y_val)), \
              class_weight=self.class_weight)#,\
            #callbacks=[early_stopping])
        else:
            X_aa3mer_train=X_aa3mer_train[:,:self.maxlen-2]
            X_ss_sparse_train=X_ss_sparse_train[:,:self.maxlen-2]
            #print X_aa3mer_train.shape, X_ss_sparse_train.shape, y_train.shape
            
            #early_stopping=EarlyStopping(monitor='val_matthews_correlation', patience=8)
            history=self.model.fit([X_aa3mer_train, X_ss_sparse_train], np_utils.to_categorical(y_train),
              batch_size=self.batch_size, \
              nb_epoch=200, \
              class_weight=self.class_weight)#,\

        ## Copy the weights and construct long Input size model based on this model.
        for key in self.model_long_input.keys():
            self.model_long_input[key].set_weights(self.model.get_weights())


       #  # summarize history for accuracy
       #  fig, (ax1, ax2, ax3) = plt.subplots(3,1)
       #  ax1.plot(history.history['acc'])
       # # ax1.plot(history.history['val_acc'])
       #  ax1.set_title('model accuracy')
       #  ax1.set_ylabel('accuracy')
       #  ax1.set_xlabel('epoch')
       #  ax1.legend(['train', 'test'], loc='upper left')
       #  # summarize history for loss
       #  ax2.plot(history.history['loss'])
       #  #ax2.plot(history.history['val_loss'])
       #  ax2.set_title('model loss')
       #  ax2.set_ylabel('loss')
       #  ax2.set_xlabel('epoch')
       #  ax2.legend(['train', 'test'], loc='upper left')
       #  # summarize history for f1
       #  ax3.plot(history.history['f1_score'])
       #  ax3.plot(history.history['val_f1_score'])
       #  ax3.set_title('f1 score')
       #  ax3.set_ylabel('f1_score')
       #  ax3.set_xlabel('epoch')
       #  ax3.legend(['train', 'test'], loc='upper left')
       #  fig.show()
       #  fig.savefig('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/DNN/SONAR_plus_DL_menthaBioPlexSTRING_ClassWeight9_training_history.pdf', format='pdf')

    def DNN_body(self, input_layer):
        ensembled_seq=Conv1D(filters=64, kernel_size=5, weights=autoEncoder_CNN.get_layer(name='conv1').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=self.CNN_trainable)(input_layer)
        #ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=CNN_trainable)(ensembled_seq)
        ensembled_seq=BatchNormalization()(ensembled_seq)
        ensembled_seq=Activation('relu')(ensembled_seq)
        ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool1')(ensembled_seq)
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        ensembled_seq=Conv1D(filters=64, kernel_size=5, weights=autoEncoder_CNN.get_layer(name='conv2').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=self.CNN_trainable)(ensembled_seq)
        #ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=CNN_trainable)(ensembled_seq)
        ensembled_seq=BatchNormalization()(ensembled_seq)
        ensembled_seq=Activation('relu')(ensembled_seq)
        ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool2')(ensembled_seq)            
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        
        #ensembled_seq=Bidirectional(LSTM(15, dropout_W=0.2, dropout_U=0.2, return_sequences=False))(ensembled_seq)
        ensembled_seq=GlobalMaxPooling1D()(ensembled_seq)
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        model=ensembled_seq
        model=Dense(21, weights=autoEncoder_Dense.get_layer(name='dense1').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense1')(model)
        model=BatchNormalization()(model)
        model=Activation('relu')(model)
        model=Dropout(self.dropout)(model)
        model=Dense(5, weights=autoEncoder_Dense.get_layer(name='dense2').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense2')(model)
        model=BatchNormalization()(model)
        model=Activation('relu')(model)
        model=Dropout(self.dropout)(model)
        # l=1
        # for n in self.denses:
        #     model=Dense(n, kernel_initializer='glorot_normal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense'+str(l))(model)
        #     model=BatchNormalization()(model)
        #     model=Activation('relu')(model)
        #     model=Dropout(self.dropout)(model)
        #     l+=1
            
        model_out=Dense(2, activation='sigmoid')(model)
        return model_out
    
    def get_model(self):
        if self.n_gpus>1:
            with tf.device(self.CPU):
                input_aa = Input((self.maxlen-2,), name='aa3mer_input')
                model_aa = Embedding(input_dim=self.BioVec_weights_add_null.shape[0], output_dim=100, input_length=self.maxlen-2, weights=[self.BioVec_weights_add_null], trainable=False, name='aa3mer_embedding')(input_aa)
                input_ss = Input((self.maxlen-2,3), name='ss_sparse_input')
                ensembled_seq = Concatenate()([model_aa, input_ss])
                model_out = self.DNN_body(ensembled_seq)
                model = Model(inputs=[input_aa, input_ss], outputs=[model_out])

            model = multi_gpu_model(model, gpus=self.n_gpus)

        else:
            input_aa = Input((self.maxlen-2,), name='aa3mer_input')
            model_aa = Embedding(input_dim=self.BioVec_weights_add_null.shape[0], output_dim=100, input_length=self.maxlen-2, weights=[self.BioVec_weights_add_null], trainable=False, name='aa3mer_embedding')(input_aa)
            input_ss = Input((self.maxlen-2,3), name='ss_sparse_input')
            ensembled_seq = Concatenate()([model_aa, input_ss])
            model_out = self.DNN_body(ensembled_seq)
            model = Model(inputs=[input_aa, input_ss], outputs=[model_out])
        # load weights
        model.compile(loss='binary_crossentropy',\
                      optimizer='Adam',\
                      metrics=['accuracy', self.f1_score])

        return model 

    def get_model_long_prot(self, prot_len):
        prot_len=int(prot_len)
        if self.n_gpus > 1:
            with tf.device(self.CPU):
                input_aa = Input((prot_len - 2,), name='aa3mer_input')
                model_aa = Embedding(input_dim=self.BioVec_weights_add_null.shape[0], output_dim=100, input_length=prot_len - 2, weights=[self.BioVec_weights_add_null], trainable=False, name='aa3mer_embedding')(input_aa)
                input_ss = Input((prot_len - 2, 3), name='ss_sparse_input')
                ensembled_seq = Concatenate()([model_aa, input_ss])
                model_out = self.DNN_body(ensembled_seq)
                model = Model(inputs=[input_aa, input_ss], outputs=[model_out])

                model = multi_gpu_model(model, gpus=self.n_gpus)

        else:
            input_aa = Input((prot_len - 2,), name='aa3mer_input')
            model_aa = Embedding(input_dim=self.BioVec_weights_add_null.shape[0], output_dim=100, input_length=prot_len - 2, weights=[self.BioVec_weights_add_null], trainable=False, name='aa3mer_embedding')(input_aa)
            input_ss = Input((prot_len - 2, 3), name='ss_sparse_input')
            ensembled_seq = Concatenate()([model_aa, input_ss])
            model_out = self.DNN_body(ensembled_seq)
            model = Model(inputs=[input_aa, input_ss], outputs=[model_out])

        # load weights
        model.compile(loss='binary_crossentropy',\
                      optimizer='Adam',\
                      metrics=['accuracy', self.f1_score])
        return model

    def predict_score(self, X_aa3mer_test, X_ss_sparse_test, X_seqlens_test):
        scores=[]
        for i in range(len(X_aa3mer_test)):
            if X_seqlens_test[i]<=self.maxlen:
                scores.append(self.model.predict([np.array([X_aa3mer_test[i][:self.maxlen-2]]), np.array([X_ss_sparse_test[i][:self.maxlen-2]])])[:,1][0])
            else:
                scores.append(self.predict_long_seq(X_aa3mer_test[i], X_ss_sparse_test[i], X_seqlens_test[i]))
                
        return np.array(scores)
     
    def predict_long_seq(self, x_aa3mer, x_ss_sparse, x_seqlen):
        input_size=int(int(x_seqlen/(self.maxlen/2))*int(self.maxlen/2)+int(self.maxlen/2))
        model_input_size=min(self.max_seq_len, input_size)
        model=self.model_long_input[model_input_size]
        if x_aa3mer.shape[0] < model_input_size: ## Used for the longest sequence when building the sequence object.
            x_aa3mer=list(x_aa3mer)+[0]*(model_input_size-x_aa3mer.shape[0])
            x_ss_sparse=list(x_ss_sparse)+[[0,0,0]]*(model_input_size-x_ss_sparse.shape[0])

        score=model.predict([np.array([x_aa3mer[:model_input_size-2]]), np.array([x_ss_sparse[:model_input_size-2]])])[:,1][0]

        return score 
    
    def save_model(self, filepath, name):
        self.model.save(os.path.join(filepath, name+'_model_param.h5'))
    
    def load_model(self, model):
        self.model=pickle.load(open(model,'r'))

    def save_model2(self, filepath, name):
        ## save the structure of the model.
        f=open(os.path.join(filepath, name+'_model_structure.json'), 'w')
        json_string = self.model.to_json()
        f.write(json_string)
        f.close()
        ## save the weights of the model.
        self.model.save_weights(os.path.join(filepath, name+'_model_weights.h5'))

    def load_model2(self, json_file, weights_file):
        f=open(json_file)
        self.model=model_from_json(f.read())
        self.model.load_weights(weights_file)
        f.close()
        for key in self.model_long_input.keys():
            self.model_long_input[key].set_weights(self.model.get_weights())

    def f1_score(self, y_true, y_pred):
        # Count positive samples.
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
        c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.
        if c3 == 0:
            return 0

        # How many selected items are relevant?
        precision = c1 / c2

        # How many relevant items are selected?
        recall = c1 / c3

        # Calculate f1_score
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score



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


def get_selected_SVM_features_1D_occlusion(RBP_uniprotID, i, k, seq, ss):
    """
    i: int, the start index of occluder
    k: int, the length of occluder.
    """
    #### Prepare occlusion for SVM
    tmp_feature_table=svm_feature_table.loc[RBP_uniprotID]
    
    ### decrease corresponding k-mer
    ## 3-mer
    #recount_kmer(RBP_uniprotID, seq, i, k, 3, tmp_feature_table)
    down_count_kmer(RBP_uniprotID, seq, i-(3-1), i+k+(3-1), 3, tmp_feature_table, mer3_selected)   
    ## 4-mer
    #recount_kmer(RBP_uniprotID, seq, i, k, 4, tmp_feature_table)
    down_count_kmer(RBP_uniprotID, seq, i-(4-1), i+k+(4-1), 4, tmp_feature_table, mer4_selected)   
    ### decrease corresponding SS-kmer
    ## 11-mer
    #recount_kmer(RBP_uniprotID, ss, i, k, 11, tmp_feature_table)
    down_count_kmer(RBP_uniprotID, ss, i-(11-1), i+k+(11-1), 11, tmp_feature_table, SSmer11_selected)   

    ## 15-mer
    #recount_kmer(RBP_uniprotID, ss, i, k, 15, tmp_feature_table)
    down_count_kmer(RBP_uniprotID, ss, i-(15-1), i+k+(15-1), 15, tmp_feature_table, SSmer15_selected)   

    ### re-run PseAAC
    seq_occ=seq[:i]+seq[(i+k):]
    tmp_paac=get_AAC_features(seq_occ) ## the output is a dictionary
    if not tmp_paac:
        raise ValueError(RBP_uniprotID+"Sequence ("+seq_occ+") gets problem in calculating PseAAC.")
    else:    
        for p in PseAAC_selected:
            tmp_feature_table[p]=tmp_paac[p]

    return tmp_feature_table

def get_selected_SVM_features_2D_occlusion(RBP_uniprotID, i1, i2, k, seq, ss):
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

def calculate_AAC(seq):
    dic = {}
    seq_len=len(seq)
    AAs=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    dic={aa:0 for aa in AAs}
    for aa in seq:
        if aa not in dic.keys():
            dic[aa]=1
        else:
            dic[aa]+=1

    return {k:v*1.0/seq_len for k, v in dic.items()}

def get_AAC_features(s, k=4):
    s=s.upper()
    s=s.replace('*','')
    s=s.replace('X','') #solve bug for those cannot be processed by propy
    s=s.replace('U','C') #solve bug for those 'U' that cannot be processed by propy
    try:
        aac=calculate_AAC(s)  #the output is a dictionary
    except Exception as e:
        print(type(e))
        print(e)
        print(s)
        return None

    return aac

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

def get_occluding_1Dheatmap_domain_stripe_SONARp_SeqOnly(RBP_file, model_DNN, model_SVM, k=5, sliding_step=1, seq_dir='/home/wjin/data2/proteins/uniport_data/canonical_seq/'):
    file=os.path.join(seq_dir,RBP_file+'.fasta')
    file_ss=os.path.join(seq_dir,RBP_file+'.spd3')
    RBP_uniprotID=RBP_file.split('.')[0]
    ## DNN features
    class_label=1
    RBP=Protein_Sequence_Input5([file], [class_label], BioVec_name_dict, max_seqlen=1500)
    RBP_aa3mer=RBP.get_aa3mer_mats()[0]#[:900]
    RBP_ss_sparse_mat = RBP.get_ss_sparse_mats2()[0]#[:900,:]
    RBP_seqlens = RBP.get_seqlens()[0]
    #RBP_aa=RBP.get_aa_mats()[0]#[:900]
    RBP_y = RBP.get_class_labels()

    ## SVM features
    print(svm_feature_table.shape)
    tmp_feature_table=svm_feature_table.loc[RBP_uniprotID]
    print(tmp_feature_table.shape)
    f=open(file)
    seq=f.read().split('\n')[1].strip(' ').strip('*')
    seq=seq.replace('X','')    
    f.close()

    tmp=pd.read_table(file_ss)
    tmp=tmp[tmp['AA']!='*']
    tmp=tmp[tmp['AA']!='X']
    ss=''.join(list(tmp['SS'])).strip(' ')
    if len(seq)!=RBP_seqlens or len(seq)!=len(ss):
        raise ValueError("Seq length is not consistent in "+RBP_uniprotID+".")

    occluded_DNN_scores=[]
    occluded_SVM_scores=[]
    occluded_ens_scores=[]
    occluded_coord=[]
    original_DNN_score=model_DNN.predict_score([RBP_aa3mer], [RBP_ss_sparse_mat], [RBP_seqlens])[0]
    original_SVM_score=model_SVM.predict_proba([np.array(tmp_feature_table)])[:,1][0]
    original_ens_score=np.array(1-get_model_fdr(scores_DNN, true_labels, original_DNN_score)*get_model_fdr(scores_SVM, true_labels, original_SVM_score))

    ## i=1
    #### occlusion for DNN 
    i=0
    RBP_aa3mer_cp=RBP_aa3mer.copy()
    RBP_ss_sparse_mat_cp=RBP_ss_sparse_mat.copy()
    RBP_aa3mer_cp[i:i+k]=0
    RBP_ss_sparse_mat_cp[i:i+k]=[0,0,0]
    y_DNN_pred = model_DNN.predict_score([RBP_aa3mer_cp], [RBP_ss_sparse_mat_cp], [RBP_seqlens])[0]
    occluded_DNN_scores.append(y_DNN_pred)
    #### Occlusion for SVM
    SVM_features0=get_selected_SVM_features_1D_occlusion(RBP_uniprotID, i, k, seq, ss)
    y_SVM_pred=model_SVM.predict_proba([np.array(SVM_features0)])[:,1][0]
    occluded_SVM_scores.append(y_SVM_pred)
    occluded_coord.append(str(i)+'-'+str(i+k-1))
    #occluded_ens_scores.append(1-get_model_fdr(scores_DNN, true_labels, y_DNN_pred)*get_model_fdr(scores_SVM, true_labels, y_SVM_pred))

    len_seq=len(seq)
    for i in range(1, len_seq-k+1, sliding_step):
        #### occlusion for DNN 
        RBP_aa3mer_cp=RBP_aa3mer.copy()
        RBP_ss_sparse_mat_cp=RBP_ss_sparse_mat.copy()
        RBP_aa3mer_cp[i:i+k]=0
        RBP_ss_sparse_mat_cp[i:i+k]=[0,0,0]
        y_DNN_pred = model_DNN.predict_score([RBP_aa3mer_cp], [RBP_ss_sparse_mat_cp], [RBP_seqlens])[0]
        occluded_DNN_scores.append(y_DNN_pred)
        #### Occlusion for SVM
        up_count_kmer_2(seq[max(0,i-3):i], SVM_features0, mer3_selected)
        down_count_kmer_2(seq[i+k-1:min(i+k+3-1, len_seq)], SVM_features0, mer3_selected)
        up_count_kmer_2(seq[max(0,i-4):i], SVM_features0, mer4_selected)
        down_count_kmer_2(seq[i+k-1:min(i+k+4-1, len_seq)], SVM_features0, mer4_selected)
        up_count_kmer_2(seq[max(0,i-11):i], SVM_features0, SSmer11_selected)
        down_count_kmer_2(seq[i+k-1:min(i+k+11-1, len_seq)], SVM_features0, SSmer11_selected)
        up_count_kmer_2(seq[max(0,i-15):i], SVM_features0, SSmer15_selected)
        down_count_kmer_2(seq[i+k-1:min(i+k+15-1, len_seq)], SVM_features0, SSmer15_selected)
        ### re-run PseAAC
        seq_occ=seq[:i]+seq[(i+k):]
        tmp_paac=get_AAC_features(seq_occ) ## the output is a dictionary
        if not tmp_paac:
            raise ValueError(RBP_uniprotID+"Sequence ("+seq_occ+") gets problem in calculating PseAAC.")
        else:    
            for p in PseAAC_selected:
                SVM_features0[p]=tmp_paac[p]


        y_SVM_pred=model_SVM.predict_proba([np.array(SVM_features0)])[:,1][0]
        occluded_SVM_scores.append(y_SVM_pred)
        occluded_coord.append(str(i)+'-'+str(i+k-1))
        #occluded_ens_scores.append(1-get_model_fdr(scores_DNN, true_labels, y_DNN_pred)*get_model_fdr(scores_SVM, true_labels, y_SVM_pred))

    return occluded_DNN_scores, occluded_SVM_scores, original_DNN_score, original_SVM_score, occluded_coord


def run(RBP_uniprotID, model_DNN, model_SVM, k=5, seq_dir='/home/wjin/data2/proteins/uniport_data/canonical_seq/'):
    path='/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/CNN_interpretation/SONAR_plus_menthaBioPlexSTRING/Occlusion_for_peptides/Occlusion_score_matrix/LysC_pep_v4_7_3_combined10RBPlist/WindowSize_'+str(k)+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    if not (os.path.exists(os.path.join(seq_dir, RBP_uniprotID+'.fasta')) and os.path.exists(os.path.join(seq_dir, RBP_uniprotID+'.spd3'))):
        return 
    if not os.path.exists(path+RBP_uniprotID+'_Occlusion_score_matrix_aac.xls'):
        print(RBP_uniprotID)
        occluded_DNN_scores, occluded_SVM_scores, original_DNN_score, original_SVM_score, occluded_coord=get_occluding_1Dheatmap_domain_stripe_SONARp_SeqOnly(RBP_uniprotID, model_DNN, model_SVM, k, seq_dir=seq_dir)
        delta_DNN=occluded_DNN_scores - original_DNN_score
        delta_SVM=occluded_SVM_scores - original_SVM_score
        #delta_ens=occluded_ens_scores - original_ens_score

        scores_df=pd.DataFrame.from_dict({'occluded_DNN_scores':occluded_DNN_scores,'occluded_SVM_scores':occluded_SVM_scores, 'original_DNN_score':original_DNN_score, 'original_SVM_score':original_SVM_score, 'delta_DNN':delta_DNN, 'delta_SVM':delta_SVM, 'occluded_coord':occluded_coord})
        scores_df.to_csv(path+RBP_uniprotID+'_Occlusion_score_matrix_aac.xls', index=True, sep='\t')

maxlen=1500
max_seq_len=15000
CNN_trainable = True
json_file='/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_models/SONAR_plus_menthaBioPlexSTRING/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_model_structure.json'
weights_file='/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_models/SONAR_plus_menthaBioPlexSTRING/DNN_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_model_weights.h5'
autoEncoder_CNN = load_model('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/DNN/model_weights/CNN_autoencoder_2CNN2MaxPool_64filter5_2poollength_first'+str(maxlen)+'aa_withRBPs_OOPSXRNAXadded.h5')
autoEncoder_Dense = load_model('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/DNN/model_weights/DenseI21_5_autoencoder_2CNN2MaxPool_64filter5_2poollength_first'+str(maxlen)+'aa_withRBPs_OOPSXRNAXadded.h5')
model_DNN=SONARp_DNN_SeqOnly(BioVec_weights_add_null, CNN_trainable=CNN_trainable, maxlen=maxlen, max_seq_len=max_seq_len, dropout=0.3, class_weight={0:1., 1:11.}, batch_size=128, val_fold=None, sliding_step=int((maxlen-2)/2), n_gpus=1)
model_DNN.load_model2(json_file, weights_file)

model_SVM=joblib.load('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_models/SONAR_plus_menthaBioPlexSTRING/SVM_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_ModelFile.pkl')
model_SVM.kernel='rbf' # To fix, sklearn conflict issue between python3 and python2 model string.

# clf_c=joblib.load('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_models/SONAR_plus_menthaBioPlexSTRING/grid_search/JointClassifier_seqOnly_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Ensemble_16_5_RBPautoEncoder_ModelFile.pkl')

# f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data/Deep_Learning_model_weights/SONAR_plus_DL_SeqOnly_ClassWeight9_TrainedWithAllShortSeq_model_param.json')
# model_SeqOnly=model_from_json(f.read())
# model_SeqOnly.load_weights('/home/wjin/projects/RBP_pred/RBP_identification/Data/Deep_Learning_model_weights/SONAR_plus_DL_SeqOnly_ClassWeight9_TrainedWithAllShortSeq_model_weights.h5')
# f.close()

# f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/SONAR+/Top200_RBPcandidates.txt')
# RBPs=f.read().split('\n')
# f.close()

## SVM_seq selected feature table
svm_feature_table=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/SVM_seq_featuresWuhanVirus.xls', index_col=0)
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
f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_AAC_chi2_alpha0.01_selected_features_From_WholeDataSet_AACname.txt')
PseAAC_selected=f.read().split('\n')
PseAAC_selected.remove('')
f.close()


k=20
t_dir='/home/wjin/data2/proteins/WuhanVirus/proteins/'
proteins=list(map(lambda x: x.split('.')[0], os.listdir(t_dir)))

#shuffle(zf_proteins)

for prot in proteins:
    run(prot, model_DNN, model_SVM, k, seq_dir=t_dir)

#pool=mp.Pool(processes=16)
#result=[pool.apply_async(run, args=(RBP, model_SeqOnly, k,)) for RBP in RBPs]
#results=[p.get() for p in result]






