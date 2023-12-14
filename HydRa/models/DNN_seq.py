#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import math
from random import shuffle
from sklearn.model_selection import StratifiedKFold
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout, Activation, Layer, InputSpec, add, Concatenate
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization, UpSampling1D
from keras import metrics
import keras.backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#from scipy import interp, stats
#from keras.layers.wrappers import TimeDistributed, Bidirectional
#from keras.layers.core import Reshape
from keras.models import load_model, model_from_json
from keras import regularizers
#import pickle
import matplotlib.pyplot as plt
import joblib
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.python.client import device_lib
#from keras.utils import multi_gpu_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

# autoEncoder_CNN=load_model('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/DNN/model_weights/CNN_autoencoder_2CNN2MaxPool_64filter5_2poollength_first1500aa_withRBPs.h5')
# autoEncoder_Dense=load_model('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/DNN/model_weights/DenseI21_5_autoencoder_2CNN2MaxPool_64filter5_2poollength_first1500aa_withRBPs.h5')
# 
# BioVec_weights=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data/protVec_100d_3grams.csv', sep='\t', header=None, index_col=0)
# BioVec_weights_add_null=np.append(np.zeros((1,100)), BioVec_weights.values, axis=0) #append a [0,0,...,0] array at the top of the matrix, which used for padding 0s.
# BioVec_weights_add_null=BioVec_weights_add_null*10
# BioVec_name_dict={}
# for i in range(1, len(BioVec_weights)+1):
#     BioVec_name_dict.update({BioVec_weights.index[i-1]:i})

class SONARp_DNN_SeqOnly_noSS:
    def __init__(self, BioVec_weights_add_null, max_seq_len=None, CNN_trainable=True, class_weight={0:1., 1:9.}, dropout=0.3, maxlen=1500, batch_size=50, val_fold=10, sliding_step=100, optimizer='Adam',n_gpus=1,CPU='/device:CPU:0', autoEncoder_CNN=None, autoEncoder_Dense=None):
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
        self.autoEncoder_CNN=autoEncoder_CNN
        self.autoEncoder_Dense=autoEncoder_Dense
        self.sliding_step=sliding_step
        self.n_gpus=n_gpus
        self.CPU=CPU
        self.model=self.get_model()
        ## set up models for long input.
        self.model_long_input = {}
        # if self.max_seq_len and self.max_seq_len>self.maxlen:
        #     input_length=int(self.maxlen+self.maxlen/2)
        #     while input_length <= (self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+(self.maxlen/2):
        #         self.model_long_input[min(input_length, self.max_seq_len)]=self.get_model_long_prot(min(input_length, self.max_seq_len))
        #         input_length+=int(self.maxlen/2)

        if self.max_seq_len and self.max_seq_len>self.maxlen:
            input_length=int(self.maxlen+self.maxlen/2)
            while input_length <= int(int(self.max_seq_len/(self.maxlen/2))*int(self.maxlen/2)+int(self.maxlen/2)):
                self.model_long_input[min(input_length, self.max_seq_len)]=self.get_model_customized_inputSize(min(input_length, self.max_seq_len))
                input_length+=int(self.maxlen/2)

    def fit(self, X_aa3mer_train, y_train):

        if self.val_fold:
            skf=StratifiedKFold(n_splits=self.val_fold)
            train, val = list(skf.split(X_aa3mer_train, y_train))[0]
            X_aa3mer_val=X_aa3mer_train[val]
            X_aa3mer_val=X_aa3mer_val[:,:self.maxlen-2]
            # X_ss_sparse_val=X_ss_sparse_train[val]
            # X_ss_sparse_val=X_ss_sparse_val[:,:self.maxlen-2]
            y_val=y_train[val]
            X_aa3mer_train=X_aa3mer_train[train]
            X_aa3mer_train=X_aa3mer_train[:,:self.maxlen-2]
            # X_ss_sparse_train=X_ss_sparse_train[train]
            # X_ss_sparse_train=X_ss_sparse_train[:,:self.maxlen-2]
            y_train=y_train[train]
            #print X_aa3mer_train.shape, X_ss_sparse_train.shape, y_train.shape
            
            #early_stopping=EarlyStopping(monitor='val_matthews_correlation', patience=8)
            history=self.model.fit([X_aa3mer_train], to_categorical(y_train),
              batch_size=self.batch_size, \
              epochs=200, \
              validation_data=([X_aa3mer_val], to_categorical(y_val)), \
              class_weight=self.class_weight)#,\
            #callbacks=[early_stopping])
        else:
            X_aa3mer_train=X_aa3mer_train[:,:self.maxlen-2]
            #X_ss_sparse_train=X_ss_sparse_train[:,:self.maxlen-2]
            #print X_aa3mer_train.shape, X_ss_sparse_train.shape, y_train.shape
            
            #early_stopping=EarlyStopping(monitor='val_matthews_correlation', patience=8)
            history=self.model.fit([X_aa3mer_train], to_categorical(y_train),
              batch_size=self.batch_size, \
              epochs=200, \
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
        if self.autoEncoder_CNN:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, weights=self.autoEncoder_CNN.get_layer(name='conv1').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=self.CNN_trainable)(input_layer)
        else:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=self.CNN_trainable)(input_layer)
        ensembled_seq=BatchNormalization()(ensembled_seq)
        ensembled_seq=Activation('relu')(ensembled_seq)
        ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool1')(ensembled_seq)
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        if self.autoEncoder_CNN:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, weights=self.autoEncoder_CNN.get_layer(name='conv2').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=self.CNN_trainable)(ensembled_seq)
        else:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=self.CNN_trainable)(ensembled_seq)
        ensembled_seq=BatchNormalization()(ensembled_seq)
        ensembled_seq=Activation('relu')(ensembled_seq)
        ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool2')(ensembled_seq)            
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        
        #ensembled_seq=Bidirectional(LSTM(15, dropout_W=0.2, dropout_U=0.2, return_sequences=False))(ensembled_seq)
        ensembled_seq=GlobalMaxPooling1D()(ensembled_seq)
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        model=ensembled_seq
        if self.autoEncoder_Dense:
            model=Dense(21, weights=self.autoEncoder_Dense.get_layer(name='dense1').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense1')(model)
        else:
            model=Dense(21, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense1')(model)
        model=BatchNormalization()(model)
        model=Activation('relu')(model)
        model=Dropout(self.dropout)(model)
        if self.autoEncoder_Dense:
            model=Dense(5, weights=self.autoEncoder_Dense.get_layer(name='dense2').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense2')(model)
        else:
            model=Dense(5, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense2')(model)
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
                # input_ss = Input((self.maxlen-2,3), name='ss_sparse_input')
                ensembled_seq = model_aa
                model_out = self.DNN_body(ensembled_seq)
                model = Model(inputs=[input_aa], outputs=[model_out])

            model = multi_gpu_model(model, gpus=self.n_gpus)

        else:
            input_aa = Input((self.maxlen-2,), name='aa3mer_input')
            model_aa = Embedding(input_dim=self.BioVec_weights_add_null.shape[0], output_dim=100, input_length=self.maxlen-2, weights=[self.BioVec_weights_add_null], trainable=False, name='aa3mer_embedding')(input_aa)
            # input_ss = Input((self.maxlen-2,3), name='ss_sparse_input')
            ensembled_seq = model_aa
            model_out = self.DNN_body(ensembled_seq)
            model = Model(inputs=[input_aa], outputs=[model_out])
        # load weights
        model.compile(loss='binary_crossentropy',\
                      optimizer='Adam',\
                      metrics=['accuracy'])

        return model 

    def get_model_long_prot(self, prot_len):
        prot_len=int(prot_len)
        if self.n_gpus > 1:
            with tf.device(self.CPU):
                input_aa = Input((prot_len - 2,), name='aa3mer_input')
                model_aa = Embedding(input_dim=self.BioVec_weights_add_null.shape[0], output_dim=100, input_length=prot_len - 2, weights=[self.BioVec_weights_add_null], trainable=False, name='aa3mer_embedding')(input_aa)
                #input_ss = Input((prot_len - 2, 3), name='ss_sparse_input')
                ensembled_seq = model_aa
                model_out = self.DNN_body(ensembled_seq)
                model = Model(inputs=[input_aa], outputs=[model_out])

                model = multi_gpu_model(model, gpus=self.n_gpus)

        else:
            input_aa = Input((prot_len - 2,), name='aa3mer_input')
            model_aa = Embedding(input_dim=self.BioVec_weights_add_null.shape[0], output_dim=100, input_length=prot_len - 2, weights=[self.BioVec_weights_add_null], trainable=False, name='aa3mer_embedding')(input_aa)
            #input_ss = Input((prot_len - 2, 3), name='ss_sparse_input')
            ensembled_seq = model_aa
            model_out = self.DNN_body(ensembled_seq)
            model = Model(inputs=[input_aa], outputs=[model_out])

        # load weights
        model.compile(loss='binary_crossentropy',\
                      optimizer='Adam',\
                      metrics=['accuracy'])
        return model

    def get_model_customized_inputSize(self, prot_len):
        prot_len=int(prot_len)
        if self.n_gpus > 1:
            with tf.device(self.CPU):
                input_aa = Input((prot_len - 2,), name='aa3mer_input')
                model_aa = Embedding(input_dim=self.BioVec_weights_add_null.shape[0], output_dim=100, input_length=prot_len - 2, weights=[self.BioVec_weights_add_null], trainable=False, name='aa3mer_embedding')(input_aa)
                model_out = self.DNN_body(model_aa)
                model = Model(inputs=[input_aa], outputs=[model_out])

                model = multi_gpu_model(model, gpus=self.n_gpus)

        else:
            input_aa = Input((prot_len - 2,), name='aa3mer_input')
            model_aa = Embedding(input_dim=self.BioVec_weights_add_null.shape[0], output_dim=100, input_length=prot_len - 2, weights=[self.BioVec_weights_add_null], trainable=False, name='aa3mer_embedding')(input_aa)
            model_out = self.DNN_body(model_aa)
            model = Model(inputs=[input_aa], outputs=[model_out])

        # load weights
        model.compile(loss='binary_crossentropy',\
                      optimizer='Adam',\
                      metrics=['accuracy'])
        return model

    def predict_score(self, X_aa3mer_test, X_seqlens_test):
        scores=[]
        for i in range(len(X_aa3mer_test)):
            if X_seqlens_test[i]<=self.maxlen:
                scores.append(self.model.predict([np.array([X_aa3mer_test[i][:self.maxlen-2]])])[:,1][0])
            else:
                scores.append(self.predict_long_seq(X_aa3mer_test[i], X_seqlens_test[i]))
                
        return np.array(scores)
     
    def predict_long_seq(self, x_aa3mer, x_seqlen):
        input_size=int(int(x_seqlen/(self.maxlen/2))*int(self.maxlen/2)+int(self.maxlen/2)) 
        input_size2=min(self.max_seq_len, input_size)
        # model=self.model_long_input[min(self.max_seq_len, input_size)]
        # score=model.predict([np.array([x_aa3mer[:input_size-2]])])[:,1][0]
        model=self.model_long_input[input_size2]
        score=model.predict([np.array([x_aa3mer[:input_size2-2]])])[:,1][0]

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
        #f=open(json_file)
        #self.model=model_from_json(f.read())
        self.model=model_from_json(json_file)
        self.model.load_weights(weights_file)
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

class SONARp_DNN_SeqOnly:
    def __init__(self, BioVec_weights_add_null, max_seq_len=None, CNN_trainable=True, class_weight={0:1., 1:9.}, dropout=0.3, maxlen=1500, batch_size=50, val_fold=10, sliding_step=100, optimizer='Adam',n_gpus=1,CPU='/device:CPU:0', autoEncoder_CNN=None, autoEncoder_Dense=None):
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
        self.autoEncoder_CNN=autoEncoder_CNN
        self.autoEncoder_Dense=autoEncoder_Dense
        self.sliding_step=sliding_step
        self.n_gpus=n_gpus
        self.CPU=CPU
        self.model=self.get_model()
        ## set up models for long input.
        self.model_long_input = {}
        if self.max_seq_len and self.max_seq_len>self.maxlen:
            input_length=int(self.maxlen+self.maxlen/2)
            while input_length <= (self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+(self.maxlen/2):
                self.model_long_input[min(input_length, self.max_seq_len)]=self.get_model_customized_inputSize(min(input_length, self.max_seq_len))
                input_length+=int(self.maxlen/2)

        # self.model_short_input = {}
        # for i in range(100, int(self.maxlen/100+1)*100+1, 100):
        #     self.model_short_input[int(i)]=self.get_model_customized_inputSize(int(i))

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
            history=self.model.fit([X_aa3mer_train, X_ss_sparse_train], to_categorical(y_train),
              batch_size=self.batch_size, \
              epochs=200, \
              validation_data=([X_aa3mer_val, X_ss_sparse_val], to_categorical(y_val)), \
              class_weight=self.class_weight)#,\
            #callbacks=[early_stopping])
        else:
            X_aa3mer_train=X_aa3mer_train[:,:self.maxlen-2]
            X_ss_sparse_train=X_ss_sparse_train[:,:self.maxlen-2]
            #print X_aa3mer_train.shape, X_ss_sparse_train.shape, y_train.shape
            
            #early_stopping=EarlyStopping(monitor='val_matthews_correlation', patience=8)
            history=self.model.fit([X_aa3mer_train, X_ss_sparse_train], to_categorical(y_train),
              batch_size=self.batch_size, \
              epochs=200, \
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

        if self.autoEncoder_CNN:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, weights=self.autoEncoder_CNN.get_layer(name='conv1').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=self.CNN_trainable)(input_layer)
        else:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=self.CNN_trainable)(input_layer)
        ensembled_seq=BatchNormalization()(ensembled_seq)
        ensembled_seq=Activation('relu')(ensembled_seq)
        ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool1')(ensembled_seq)
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        if self.autoEncoder_CNN:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, weights=self.autoEncoder_CNN.get_layer(name='conv2').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=self.CNN_trainable)(ensembled_seq)
        else:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=self.CNN_trainable)(ensembled_seq)
        ensembled_seq=BatchNormalization()(ensembled_seq)
        ensembled_seq=Activation('relu')(ensembled_seq)
        ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool2')(ensembled_seq)            
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        
        #ensembled_seq=Bidirectional(LSTM(15, dropout_W=0.2, dropout_U=0.2, return_sequences=False))(ensembled_seq)
        ensembled_seq=GlobalMaxPooling1D()(ensembled_seq)
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        model=ensembled_seq
        if self.autoEncoder_Dense:
            model=Dense(21, weights=self.autoEncoder_Dense.get_layer(name='dense1').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense1')(model)
        else:
            model=Dense(21, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense1')(model)
        model=BatchNormalization()(model)
        model=Activation('relu')(model)
        model=Dropout(self.dropout)(model)
        if self.autoEncoder_Dense:
            model=Dense(5, weights=self.autoEncoder_Dense.get_layer(name='dense2').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense2')(model)
        else:
            model=Dense(5, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense2')(model)
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
                      metrics=['accuracy'])

        return model 

    def get_model_customized_inputSize(self, prot_len):
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
                      metrics=['accuracy'])
        return model


    def predict_score(self, X_aa3mer_test, X_ss_sparse_test, X_seqlens_test):
        scores=[]
        for i in range(len(X_aa3mer_test)):
            # if X_seqlens_test[i]<=self.maxlen/2:
            #     scores.append(self.predict_short_seq(X_aa3mer_test[i], X_ss_sparse_test[i], X_seqlens_test[i]))
            if X_seqlens_test[i]<=self.maxlen:
                scores.append(self.model.predict([np.array([X_aa3mer_test[i][:self.maxlen-2]]), np.array([X_ss_sparse_test[i][:self.maxlen-2]])])[:,1][0])
            else:
                scores.append(self.predict_long_seq(X_aa3mer_test[i], X_ss_sparse_test[i], X_seqlens_test[i]))
                
        return np.array(scores)
    
    def predict_long_seq(self, x_aa3mer, x_ss_sparse, x_seqlen):
        input_size=int(int(x_seqlen/(self.maxlen/2))*int(self.maxlen/2)+int(self.maxlen/2))
        input_size2=min(self.max_seq_len, input_size)
        #model=self.model_long_input[input_size]     
        #score=model.predict([np.array([x_aa3mer[:input_size-2]]), np.array([x_ss_sparse[:input_size-2]])])[:,1][0]
        model=self.model_long_input[input_size2] 
        score=model.predict([np.array([x_aa3mer[:input_size2-2]]), np.array([x_ss_sparse[:input_size2-2]])])[:,1][0]
        
        return score 

    # def predict_short_seq(self, x_aa3mer, x_ss_sparse, x_seqlen):
    #     input_size=int(int(x_seqlen-1)/100+1)*100
    #     model=self.model_short_input[min(self.max_seq_len, input_size)]        
    #     score=model.predict([np.array([x_aa3mer[:input_size-2]]), np.array([x_ss_sparse[:input_size-2]])])[:,1][0]
    #     return score 

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
        # f=open(json_file)
        # self.model=model_from_json(f.read())
        self.model=model_from_json(json_file)
        self.model.load_weights(weights_file)
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

class SONARp_DNN_SeqOnly_2:
    def __init__(self, BioVec_weights_add_null, max_seq_len=None, CNN_trainable=True, class_weight={0:1., 1:9.}, dropout=0.3, maxlen=1500, batch_size=50, val_fold=10, sliding_step=100, optimizer='Adam',n_gpus=1,CPU='/device:CPU:0', autoEncoder_CNN=None, autoEncoder_Dense=None):
        """
        Difference from previous version: Also generate input-size-customized model for short proteins.
        Difference from previous version: use customized input size when doing prediction. Fitting process is kept same.
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
        self.autoEncoder_CNN=autoEncoder_CNN
        self.autoEncoder_Dense=autoEncoder_Dense
        self.sliding_step=sliding_step
        self.n_gpus=n_gpus
        self.CPU=CPU
        self.model=self.get_model()
        ## set up models for long input.
        self.model_long_input = {}
        if self.max_seq_len and self.max_seq_len>self.maxlen:
            input_length=int(self.maxlen+self.maxlen/2)
            while input_length <= (self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+(self.maxlen/2):
                self.model_long_input[min(input_length, self.max_seq_len)]=self.get_model_customized_inputSize(min(input_length, self.max_seq_len))
                input_length+=int(self.maxlen/2)

        self.model_short_input = {}
        for i in range(100, int(self.maxlen/100+1)*100+1, 100):
            self.model_short_input[int(i)]=self.get_model_customized_inputSize(int(i))

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
            history=self.model.fit([X_aa3mer_train, X_ss_sparse_train], to_categorical(y_train),
              batch_size=self.batch_size, \
              epochs=200, \
              validation_data=([X_aa3mer_val, X_ss_sparse_val], to_categorical(y_val)), \
              class_weight=self.class_weight)#,\
            #callbacks=[early_stopping])
        else:
            X_aa3mer_train=X_aa3mer_train[:,:self.maxlen-2]
            X_ss_sparse_train=X_ss_sparse_train[:,:self.maxlen-2]
            #print X_aa3mer_train.shape, X_ss_sparse_train.shape, y_train.shape
            
            #early_stopping=EarlyStopping(monitor='val_matthews_correlation', patience=8)
            history=self.model.fit([X_aa3mer_train, X_ss_sparse_train], to_categorical(y_train),
              batch_size=self.batch_size, \
              epochs=200, \
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
        if self.autoEncoder_CNN:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, weights=self.autoEncoder_CNN.get_layer(name='conv1').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=self.CNN_trainable)(input_layer)
        else:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=self.CNN_trainable)(input_layer)
        ensembled_seq=BatchNormalization()(ensembled_seq)
        ensembled_seq=Activation('relu')(ensembled_seq)
        ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool1')(ensembled_seq)
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        if self.autoEncoder_CNN:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, weights=self.autoEncoder_CNN.get_layer(name='conv2').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=self.CNN_trainable)(ensembled_seq)
        else:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=self.CNN_trainable)(ensembled_seq)
        ensembled_seq=BatchNormalization()(ensembled_seq)
        ensembled_seq=Activation('relu')(ensembled_seq)
        ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool2')(ensembled_seq)            
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        
        #ensembled_seq=Bidirectional(LSTM(15, dropout_W=0.2, dropout_U=0.2, return_sequences=False))(ensembled_seq)
        ensembled_seq=GlobalMaxPooling1D()(ensembled_seq)
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        model=ensembled_seq
        if self.autoEncoder_Dense:
            model=Dense(21, weights=self.autoEncoder_Dense.get_layer(name='dense1').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense1')(model)
        else:
            model=Dense(21, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense1')(model)
        model=BatchNormalization()(model)
        model=Activation('relu')(model)
        model=Dropout(self.dropout)(model)
        if self.autoEncoder_Dense:
            model=Dense(5, weights=self.autoEncoder_Dense.get_layer(name='dense2').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense2')(model)
        else:
            model=Dense(5, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense2')(model)
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
                      metrics=['accuracy'])

        return model 

    def get_model_customized_inputSize(self, prot_len):
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
                      metrics=['accuracy'])
        return model


    def predict_score(self, X_aa3mer_test, X_ss_sparse_test, X_seqlens_test):
        scores=[]
        for i in range(len(X_aa3mer_test)):
            if X_seqlens_test[i]<=self.maxlen/2:
                scores.append(self.predict_short_seq(X_aa3mer_test[i], X_ss_sparse_test[i], X_seqlens_test[i]))
            elif X_seqlens_test[i]<=self.maxlen:
                scores.append(self.model.predict([np.array([X_aa3mer_test[i][:self.maxlen-2]]), np.array([X_ss_sparse_test[i][:self.maxlen-2]])])[:,1][0])
            else:
                scores.append(self.predict_long_seq(X_aa3mer_test[i], X_ss_sparse_test[i], X_seqlens_test[i]))
                
        return np.array(scores)
    
    def predict_long_seq(self, x_aa3mer, x_ss_sparse, x_seqlen):
        input_size=int(int(x_seqlen/(self.maxlen/2))*int(self.maxlen/2)+int(self.maxlen/2))  
        model=self.model_long_input[min(self.max_seq_len, input_size)]        
        score=model.predict([np.array([x_aa3mer[:input_size-2]]), np.array([x_ss_sparse[:input_size-2]])])[:,1][0]
        return score 

    def predict_short_seq(self, x_aa3mer, x_ss_sparse, x_seqlen):
        input_size=int(int(x_seqlen-1)/100+1)*100
        model=self.model_short_input[min(self.max_seq_len, input_size)]        
        score=model.predict([np.array([x_aa3mer[:input_size-2]]), np.array([x_ss_sparse[:input_size-2]])])[:,1][0]
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
        # f=open(json_file)
        # self.model=model_from_json(f.read())
        self.model=model_from_json(json_file)
        self.model.load_weights(weights_file)
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

class SONARp_DNN_SeqOnly_bk:
    def __init__(self, BioVec_weights_add_null, CNN_trainable=True, class_weight={0:1., 1:9.}, dropout=0.3, maxlen=1000, batch_size=50, val_fold=10, sliding_step=100, autoEncoder_CNN=None, autoEncoder_Dense=None):
        """
        BioVec_weights_add_null: numpy.array, embedding weights matrix for aa triplets.
        CNN_trainable: Boolean, whether allow weights updated in CNN section during model training.
        class_weights: Dictionary, weights for different class. Used in loss function of DNN training.
        dropout: float/double, the dropout used in DNN model.
        maxlen: int, the maximum length of the sequence that will not activate sliding window mode.
        batch_size: int, the batch size in DNN model training.
        val_fold: int, the k for k-fold split of the training set to training subset and validation set. if None, there is not validation set.
        sliding_step: int, the step length in sliding window mode, which is used for long proteins sequence (longer than maxlen).
        """
        self.class_weight=class_weight
        self.maxlen=maxlen
        self.dropout=dropout
        self.val_fold=val_fold
        self.CNN_trainable=CNN_trainable
        self.batch_size=batch_size
        self.class_weight=class_weight
        self.BioVec_weights_add_null=BioVec_weights_add_null
        self.autoEncoder_CNN=autoEncoder_CNN
        self.autoEncoder_Dense=autoEncoder_Dense
        self.sliding_step=sliding_step
        self.model=self.get_model(CNN_trainable=CNN_trainable)


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
            history=self.model.fit([X_aa3mer_train, X_ss_sparse_train], to_categorical(y_train),
              batch_size=self.batch_size, \
              epochs=200, \
              validation_data=([X_aa3mer_val, X_ss_sparse_val], to_categorical(y_val)), \
              class_weight=self.class_weight)#,\
            #callbacks=[early_stopping])
        else:
            X_aa3mer_train=X_aa3mer_train[:,:self.maxlen-2]
            X_ss_sparse_train=X_ss_sparse_train[:,:self.maxlen-2]
            #print X_aa3mer_train.shape, X_ss_sparse_train.shape, y_train.shape
            
            #early_stopping=EarlyStopping(monitor='val_matthews_correlation', patience=8)
            history=self.model.fit([X_aa3mer_train, X_ss_sparse_train], to_categorical(y_train),
              batch_size=self.batch_size, \
              epochs=200, \
              class_weight=self.class_weight)#,\

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

    
    def get_model(self, CNN_trainable=False):
        input_aa=Input((self.maxlen-2,), name='aa3mer_input')
        model_aa=Embedding(input_dim=self.BioVec_weights_add_null.shape[0], output_dim=100, input_length=self.maxlen-2, weights=[self.BioVec_weights_add_null], trainable=False, name='aa3mer_embedding')(input_aa)
        #input_ss=Input((self.maxlen,), name='ss3mer_input')
        input_ss=Input((self.maxlen-2,3), name='ss_sparse_input')
        #model_ss=Embedding(29, 10, input_length=self.maxlen, init='glorot_normal', name='ss3mer_embedding', trainable=CNN_trainable)(input_ss) #, dropout=0.2
        
        #ensembled_seq=merge([model_aa, input_ss], mode='concat', concat_axis=2)
        ensembled_seq=Concatenate()([model_aa, input_ss])
        if self.autoEncoder_CNN:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, weights=self.autoEncoder_CNN.get_layer(name='conv1').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=CNN_trainable)(ensembled_seq)
        else:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=CNN_trainable)(ensembled_seq)
        ensembled_seq=BatchNormalization()(ensembled_seq)
        ensembled_seq=Activation('relu')(ensembled_seq)
        ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool1')(ensembled_seq)
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        if self.autoEncoder_CNN:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, weights=self.autoEncoder_CNN.get_layer(name='conv2').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=CNN_trainable)(ensembled_seq)
        else:
            ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=CNN_trainable)(ensembled_seq)
        ensembled_seq=BatchNormalization()(ensembled_seq)
        ensembled_seq=Activation('relu')(ensembled_seq)
        ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool2')(ensembled_seq)            
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)

        
        #ensembled_seq=Bidirectional(LSTM(15, dropout_W=0.2, dropout_U=0.2, return_sequences=False))(ensembled_seq)
        ensembled_seq=GlobalMaxPooling1D()(ensembled_seq)
        ensembled_seq=Dropout(self.dropout)(ensembled_seq)
        model=ensembled_seq
        if self.autoEncoder_Dense:
            model=Dense(21, weights=self.autoEncoder_Dense.get_layer(name='dense1').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense1')(model)
        else:
            model=Dense(21, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense1')(model)
        model=BatchNormalization()(model)
        model=Activation('relu')(model)
        model=Dropout(self.dropout)(model)
        if self.autoEncoder_Dense:
            model=Dense(5, weights=self.autoEncoder_Dense.get_layer(name='dense2').get_weights(), kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense2')(model)
        else:
            model=Dense(5, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), name='dense2')(model)
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

        model_out=Dense(2, activation='sigmoid',name='dense_out')(model)
        print(model) 

        #model=Model(input=[input_aa, input_ss, PPI_input, PseAAC_input], output=[model_out])
        model=Model(input=[input_aa, input_ss], output=[model_out])
        # load weights
        model.compile(loss='binary_crossentropy',\
                      optimizer='Adam',\
                      metrics=['accuracy'])

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
        start=0
        scores=[]
        while start+self.maxlen<=x_seqlen:  
            scores.append(self.model.predict([np.array([x_aa3mer[start:(start+self.maxlen-2)]]), np.array([x_ss_sparse[start:(start+self.maxlen-2)]])])[:,1][0])
            start+=self.sliding_step
            
        if start<x_seqlen-2 and x_seqlen-2-start>=50:   # To be optimized. 50 may not be the best choice.
            scores.append(self.model.predict([np.array([list(x_aa3mer[start:x_seqlen-2])+[0]*(self.maxlen-(x_seqlen-start))]), np.array([list(x_ss_sparse[start:x_seqlen-2])+[[0,0,0]]*(self.maxlen-(x_seqlen-start))])])[:,1][0])

        return max(scores)
    
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
        # f=open(json_file)
        # self.model=model_from_json(f.read())
        self.model=model_from_json(json_file)
        self.model.load_weights(weights_file)

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

def autoEncoder_training(out_dir, BioVec_weights_add_null, X_aa3mer, X_ss_sparse=None, noSS=False, model_name='Model', maxlen=1500, batch_size=128, epochs=300, optimizer='nadam', loss='mse'):
    if (X_ss_sparse==None) and (noSS==False):
        raise ValueError('Secondary structure data should be provided.')

    input_aa=Input((maxlen,), name='aa3mer_input')
    model_aa=Embedding(input_dim=BioVec_weights_add_null.shape[0], output_dim=100, input_length=1500, weights=[BioVec_weights_add_null*10], trainable=False, name='aa3mer_embedding')(input_aa)
    if noSS:
        prepare=Model(inputs=[input_aa], outputs=[model_aa])
        prepare.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        emsembled_train=prepare.predict([X_aa3mer], batch_size=1)
    else:
        input_ss=Input((maxlen,3), name='ss3mer_input')
        ensembled_seq=Concatenate()([model_aa, input_ss])
        prepare=Model(input=[input_aa, input_ss], output=[ensembled_seq])
        prepare.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        emsembled_train=prepare.predict([X_aa3mer, X_ss_sparse], batch_size=1)

    if noSS:
        input_seq=Input(shape=(maxlen, 100))
    else:
        input_seq=Input(shape=(maxlen, 103))
    ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv1', trainable=True)(input_seq)
    ensembled_seq=BatchNormalization()(ensembled_seq)
    ensembled_seq=Activation('relu')(ensembled_seq)
    print(ensembled_seq)
    ensembled_seq=MaxPooling1D(pool_size=2, padding='same', name='maxpool1')(ensembled_seq)
    ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv2', trainable=True)(ensembled_seq)
    ensembled_seq=BatchNormalization()(ensembled_seq)
    ensembled_seq=Activation('relu')(ensembled_seq)
    encoded=MaxPooling1D(pool_size=2, padding='same', name='maxpool2')(ensembled_seq)

    ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv3', trainable=True)(encoded)
    ensembled_seq=BatchNormalization()(ensembled_seq)
    ensembled_seq=Activation('relu')(ensembled_seq)
    ensembled_seq=UpSampling1D(size=2, name='upsampling1')(ensembled_seq)
    #ensembled_seq=BatchNormalization()(ensembled_seq)
    print(ensembled_seq)
    ensembled_seq=Conv1D(filters=64, kernel_size=5, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', name='conv4', trainable=True)(ensembled_seq)
    ensembled_seq=BatchNormalization()(ensembled_seq)
    ensembled_seq=Activation('relu')(ensembled_seq)
    ensembled_seq=UpSampling1D(size=2, name='upsampling2')(ensembled_seq)
    print(ensembled_seq)
    if noSS:
        decoded=Conv1D(filters=100, kernel_size=5, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', activation='tanh', name='conv5', trainable=True)(ensembled_seq)
    else:
        decoded=Conv1D(filters=103, kernel_size=5, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), padding='same', activation='tanh', name='conv5', trainable=True)(ensembled_seq)

    #early_stopping=EarlyStopping(monitor='acc', patience=2)
    autoencoder_CNN=Model(inputs=[input_seq], outputs=[decoded])
    #autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    autoencoder_CNN.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history=autoencoder_CNN.fit(emsembled_train, emsembled_train, epochs=300, batch_size=batch_size, shuffle=True)#, callbacks=[early_stopping])
    autoencoder_CNN.save(os.path.join(out_dir, model_name+'_CNN_autoencoder_withRBPs.h5'))

    gbmax_encoded=GlobalMaxPooling1D()(encoded)
    model_dense=Model(inputs=[input_seq], outputs=[gbmax_encoded])
    model_dense.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    CNN_out_train=model_dense.predict(emsembled_train)
    dense_input=Input((64,), name='DenseI_input')
    encoded_dense=Dense(21, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), activation='relu', name='dense1', trainable=True)(dense_input)
    encoded_dense=Dense(5, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), activation='relu', name='dense2', trainable=True)(encoded_dense)

    decoded_dense=Dense(21, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), activation='relu', name='dense3', trainable=True)(encoded_dense)
    decoded_dense=Dense(64, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), activation='tanh', name='dense4', trainable=True)(decoded_dense)
    autoencoder_dense=Model(inputs=[dense_input], outputs=[decoded_dense])
    autoencoder_dense.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    history=autoencoder_dense.fit(CNN_out_train, CNN_out_train, epochs=300, batch_size=64, shuffle=True)#, callbacks=[early_stopping])
    autoencoder_dense.save(os.path.join(out_dir, model_name+'_DenseLayer_autoencoder_withRBPs.h5'))
    
    return autoencoder_CNN, autoencoder_dense

