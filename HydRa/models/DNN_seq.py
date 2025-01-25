#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import math
from random import shuffle
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, BatchNormalization, Activation,
    MaxPooling1D, GlobalMaxPooling1D, Embedding
)
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import pickle

class SONARp_DNN_SeqOnly_noSS:
    def __init__(
        self,
        BioVec_weights_add_null,
        max_seq_len=None,
        CNN_trainable=True,
        class_weight={0:1., 1:9.},
        dropout=0.3,
        maxlen=1500,
        batch_size=50,
        val_fold=10,
        sliding_step=100,
        optimizer='Adam',
        n_gpus=1,
        CPU='/device:CPU:0',
        autoEncoder_CNN=None,
        autoEncoder_Dense=None
    ):
        """
        BioVec_weights_add_null: numpy.array, embedding weights matrix for aa triplets.
        CNN_trainable: Boolean, whether allow weights updated in CNN section during model training.
        class_weights: Dictionary, weights for different class. Used in loss function of DNN training.
        dropout: float/double, the dropout used in DNN model.
        maxlen: int, the maximum length of the sequence that will not activate sliding window mode.
        batch_size: int, the batch size in DNN model training.
        val_fold: int, the k for k-fold split of the training set to training subset and validation set.
        sliding_step: int, the step length in sliding window mode for long proteins.
        """
        self.class_weight = class_weight
        self.maxlen = int(maxlen)
        self.max_seq_len = max_seq_len
        if self.max_seq_len:
            self.max_seq_len = int(self.max_seq_len)
        self.dropout = dropout
        self.val_fold = val_fold
        self.CNN_trainable = CNN_trainable
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.BioVec_weights_add_null = BioVec_weights_add_null
        self.autoEncoder_CNN = autoEncoder_CNN
        self.autoEncoder_Dense = autoEncoder_Dense
        self.sliding_step = sliding_step
        self.n_gpus = n_gpus
        self.CPU = CPU
        self.model = self.get_model()
        self.model_long_input = {}

        if self.max_seq_len and self.max_seq_len > self.maxlen:
            input_length = int(self.maxlen + self.maxlen/2)
            while input_length <= int(int(self.max_seq_len/(self.maxlen/2))*int(self.maxlen/2) + int(self.maxlen/2)):
                self.model_long_input[min(input_length, self.max_seq_len)] = \
                    self.get_model_customized_inputSize(min(input_length, self.max_seq_len))
                input_length += int(self.maxlen/2)

    def fit(self, X_aa3mer_train, y_train):
        if self.val_fold:
            skf = StratifiedKFold(n_splits=self.val_fold, shuffle=True, random_state=42)
            train, val = list(skf.split(X_aa3mer_train, y_train))[0]

            X_aa3mer_val = X_aa3mer_train[val][:,:self.maxlen-2]
            y_val = y_train[val]
            X_aa3mer_train = X_aa3mer_train[train][:,:self.maxlen-2]
            y_train = y_train[train]

            history = self.model.fit(
                X_aa3mer_train,
                to_categorical(y_train),
                batch_size=self.batch_size,
                epochs=200,
                validation_data=(X_aa3mer_val, to_categorical(y_val)),
                class_weight=self.class_weight
            )
        else:
            X_aa3mer_train = X_aa3mer_train[:,:self.maxlen-2]
            history = self.model.fit(
                X_aa3mer_train,
                to_categorical(y_train),
                batch_size=self.batch_size,
                epochs=200,
                class_weight=self.class_weight
            )

        # Copy weights to long input models
        for model in self.model_long_input.values():
            model.set_weights(self.model.get_weights())

    def DNN_body(self, input_layer):
        conv_params = {
            'filters': 64,
            'kernel_size': 5,
            'kernel_regularizer': regularizers.l2(0.01),
            'bias_regularizer': regularizers.l2(0.01),
            'padding': 'same'
        }

        # First Conv1D block
        if self.autoEncoder_CNN:
            weights = self.autoEncoder_CNN.get_layer('conv1').get_weights()
            ensembled_seq = Conv1D(**conv_params, weights=weights, name='conv1',
                                 trainable=self.CNN_trainable)(input_layer)
        else:
            ensembled_seq = Conv1D(**conv_params, name='conv1',
                                 trainable=self.CNN_trainable)(input_layer)

        ensembled_seq = BatchNormalization()(ensembled_seq)
        ensembled_seq = Activation('relu')(ensembled_seq)
        ensembled_seq = MaxPooling1D(pool_size=2, padding='same', name='maxpool1')(ensembled_seq)
        ensembled_seq = Dropout(self.dropout)(ensembled_seq)

        # Second Conv1D block
        if self.autoEncoder_CNN:
            weights = self.autoEncoder_CNN.get_layer('conv2').get_weights()
            ensembled_seq = Conv1D(**conv_params, weights=weights, name='conv2',
                                 trainable=self.CNN_trainable)(ensembled_seq)
        else:
            ensembled_seq = Conv1D(**conv_params, name='conv2',
                                 trainable=self.CNN_trainable)(ensembled_seq)

        ensembled_seq = BatchNormalization()(ensembled_seq)
        ensembled_seq = Activation('relu')(ensembled_seq)
        ensembled_seq = MaxPooling1D(pool_size=2, padding='same', name='maxpool2')(ensembled_seq)
        ensembled_seq = Dropout(self.dropout)(ensembled_seq)

        # Global pooling and dense layers
        model = GlobalMaxPooling1D()(ensembled_seq)
        model = Dropout(self.dropout)(model)

        dense_params = {
            'kernel_regularizer': regularizers.l2(0.01),
            'bias_regularizer': regularizers.l2(0.01)
        }

        # First dense block
        if self.autoEncoder_Dense:
            weights = self.autoEncoder_Dense.get_layer('dense1').get_weights()
            model = Dense(21, weights=weights, name='dense1', **dense_params)(model)
        else:
            model = Dense(21, name='dense1', **dense_params)(model)

        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(self.dropout)(model)

        # Second dense block
        if self.autoEncoder_Dense:
            weights = self.autoEncoder_Dense.get_layer('dense2').get_weights()
            model = Dense(5, weights=weights, name='dense2', **dense_params)(model)
        else:
            model = Dense(5, name='dense2', **dense_params)(model)

        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(self.dropout)(model)

        return Dense(2, activation='sigmoid')(model)

    def get_model(self):
        strategy = None
        if self.n_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()

        if strategy:
            with strategy.scope():
                return self._build_model()
        else:
            return self._build_model()

    def _build_model(self):
        input_aa = Input((self.maxlen-2,), name='aa3mer_input')
        model_aa = Embedding(
            input_dim=self.BioVec_weights_add_null.shape[0],
            output_dim=100,
            input_length=self.maxlen-2,
            weights=[self.BioVec_weights_add_null],
            trainable=False,
            name='aa3mer_embedding'
        )(input_aa)

        model_out = self.DNN_body(model_aa)
        model = Model(inputs=[input_aa], outputs=[model_out])

        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        return model

    def get_model_customized_inputSize(self, prot_len):
        prot_len = int(prot_len)
        strategy = None
        if self.n_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()

        if strategy:
            with strategy.scope():
                return self._build_custom_model(prot_len)
        else:
            return self._build_custom_model(prot_len)

    def _build_custom_model(self, prot_len):
        input_aa = Input((prot_len - 2,), name='aa3mer_input')
        model_aa = Embedding(
            input_dim=self.BioVec_weights_add_null.shape[0],
            output_dim=100,
            input_length=prot_len - 2,
            weights=[self.BioVec_weights_add_null],
            trainable=False,
            name='aa3mer_embedding'
        )(input_aa)

        model_out = self.DNN_body(model_aa)
        model = Model(inputs=[input_aa], outputs=[model_out])

        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        return model

    def predict_score(self, X_aa3mer_test, X_seqlens_test):
        scores = []
        for i in range(len(X_aa3mer_test)):
            if X_seqlens_test[i] <= self.maxlen:
                scores.append(
                    self.model.predict(
                        np.array([X_aa3mer_test[i][:self.maxlen-2]]),
                        verbose=0
                    )[:,1][0]
                )
            else:
                scores.append(
                    self.predict_long_seq(X_aa3mer_test[i], X_seqlens_test[i])
                )
        return np.array(scores)

    def predict_long_seq(self, x_aa3mer, x_seqlen):
        input_size = int(int(x_seqlen/(self.maxlen/2))*int(self.maxlen/2)+int(self.maxlen/2))
        input_size2 = min(self.max_seq_len, input_size)
        model = self.model_long_input[input_size2]
        score = model.predict(
            np.array([x_aa3mer[:input_size2-2]]),
            verbose=0
        )[:,1][0]
        return score

    def save_model(self, filepath, name):
        self.model.save(os.path.join(filepath, f'{name}_model_param.h5'))

    def save_model2(self, filepath, name):
        # Save model architecture
        config = self.model.to_json()
        with open(os.path.join(filepath, f'{name}_model_structure.json'), 'w') as f:
            f.write(config)
        # Save weights
        self.model.save_weights(os.path.join(filepath, f'{name}_model_weights.h5'))

    def load_model2(self, json_file, weights_file):
        #with open(json_file, 'r') as f:
        #    json_config = f.read()
        json_config = json_file
        self.model = tf.keras.models.model_from_json(json_config)
        self.model.load_weights(weights_file)

        # Update weights for long input models
        for model in self.model_long_input.values():
            model.set_weights(self.model.get_weights())
    
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
    def __init__(self, BioVec_weights_add_null, max_seq_len=None, CNN_trainable=True, 
                 class_weight={0:1., 1:9.}, dropout=0.3, maxlen=1500, batch_size=50, 
                 val_fold=10, sliding_step=100, optimizer='Adam', n_gpus=1, 
                 CPU='/device:CPU:0', autoEncoder_CNN=None, autoEncoder_Dense=None):
        """
        BioVec_weights_add_null: numpy.array, embedding weights matrix for aa triplets.
        CNN_trainable: Boolean, whether allow weights updated in CNN section during model training.
        class_weights: Dictionary, weights for different class. Used in loss function of DNN training.
        dropout: float/double, the dropout used in DNN model.
        maxlen: int, the maximum length of the sequence that will not activate sliding window mode.
        batch_size: int, the batch size in DNN model training.
        val_fold: int, the k for k-fold split of the training set to training subset and validation set.
        sliding_step: int, the step length in sliding window mode for long proteins sequence.
        """
        self.class_weight = class_weight
        self.maxlen = int(maxlen)
        self.max_seq_len = max_seq_len
        if self.max_seq_len:
            self.max_seq_len = int(self.max_seq_len)
        self.dropout = dropout
        self.val_fold = val_fold
        self.CNN_trainable = CNN_trainable
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.BioVec_weights_add_null = BioVec_weights_add_null
        self.autoEncoder_CNN = autoEncoder_CNN
        self.autoEncoder_Dense = autoEncoder_Dense
        self.sliding_step = sliding_step
        self.n_gpus = n_gpus
        self.CPU = CPU
        self.strategy = tf.distribute.MirroredStrategy() if n_gpus > 1 else None
        self.model = self.get_model()
        
        # Set up models for long input
        self.model_long_input = {}
        if self.max_seq_len and self.max_seq_len > self.maxlen:
            input_length = int(self.maxlen + self.maxlen/2)
            while input_length <= (self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+(self.maxlen/2):
                self.model_long_input[min(input_length, self.max_seq_len)] = \
                    self.get_model_customized_inputSize(min(input_length, self.max_seq_len))
                input_length += int(self.maxlen/2)

    def DNN_body(self, input_layer):
        # First Conv1D block
        if self.autoEncoder_CNN:
            weights = self.autoEncoder_CNN.get_layer(name='conv1').get_weights()
            ensembled_seq = Conv1D(
                filters=64, kernel_size=5, weights=weights,
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                padding='same', name='conv1', trainable=self.CNN_trainable
            )(input_layer)
        else:
            ensembled_seq = Conv1D(
                filters=64, kernel_size=5,
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                padding='same', name='conv1', trainable=self.CNN_trainable
            )(input_layer)
        
        ensembled_seq = BatchNormalization()(ensembled_seq)
        ensembled_seq = Activation('relu')(ensembled_seq)
        ensembled_seq = MaxPooling1D(pool_size=2, padding='same', name='maxpool1')(ensembled_seq)
        ensembled_seq = Dropout(self.dropout)(ensembled_seq)
        
        # Second Conv1D block
        if self.autoEncoder_CNN:
            weights = self.autoEncoder_CNN.get_layer(name='conv2').get_weights()
            ensembled_seq = Conv1D(
                filters=64, kernel_size=5, weights=weights,
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                padding='same', name='conv2', trainable=self.CNN_trainable
            )(ensembled_seq)
        else:
            ensembled_seq = Conv1D(
                filters=64, kernel_size=5,
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                padding='same', name='conv2', trainable=self.CNN_trainable
            )(ensembled_seq)
            
        ensembled_seq = BatchNormalization()(ensembled_seq)
        ensembled_seq = Activation('relu')(ensembled_seq)
        ensembled_seq = MaxPooling1D(pool_size=2, padding='same', name='maxpool2')(ensembled_seq)
        ensembled_seq = Dropout(self.dropout)(ensembled_seq)
        
        # Global pooling and dense layers
        ensembled_seq = GlobalMaxPooling1D()(ensembled_seq)
        ensembled_seq = Dropout(self.dropout)(ensembled_seq)
        
        # Dense layers
        if self.autoEncoder_Dense:
            model = Dense(
                21, weights=self.autoEncoder_Dense.get_layer(name='dense1').get_weights(),
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense1'
            )(ensembled_seq)
        else:
            model = Dense(
                21, kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01), name='dense1'
            )(ensembled_seq)
            
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(self.dropout)(model)
        
        if self.autoEncoder_Dense:
            model = Dense(
                5, weights=self.autoEncoder_Dense.get_layer(name='dense2').get_weights(),
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense2'
            )(model)
        else:
            model = Dense(
                5, kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01), name='dense2'
            )(model)
            
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(self.dropout)(model)
        
        model_out = Dense(2, activation='sigmoid')(model)
        return model_out

    def get_model(self):
        if self.n_gpus > 1:
            with self.strategy.scope():
                return self._build_model()
        else:
            return self._build_model()
    
    def _build_model(self):
        input_aa = Input((self.maxlen-2,), name='aa3mer_input')
        model_aa = Embedding(
            input_dim=self.BioVec_weights_add_null.shape[0],
            output_dim=100,
            input_length=self.maxlen-2,
            weights=[self.BioVec_weights_add_null],
            trainable=False,
            name='aa3mer_embedding'
        )(input_aa)
        
        input_ss = Input((self.maxlen-2,3), name='ss_sparse_input')
        ensembled_seq = Concatenate()([model_aa, input_ss])
        model_out = self.DNN_body(ensembled_seq)
        model = Model(inputs=[input_aa, input_ss], outputs=[model_out])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )
        return model

    def get_model_customized_inputSize(self, prot_len):
        prot_len = int(prot_len)
        if self.n_gpus > 1:
            with self.strategy.scope():
                return self._build_model_customized(prot_len)
        else:
            return self._build_model_customized(prot_len)
    
    def _build_model_customized(self, prot_len):
        input_aa = Input((prot_len-2,), name='aa3mer_input')
        model_aa = Embedding(
            input_dim=self.BioVec_weights_add_null.shape[0],
            output_dim=100,
            input_length=prot_len-2,
            weights=[self.BioVec_weights_add_null],
            trainable=False,
            name='aa3mer_embedding'
        )(input_aa)
        
        input_ss = Input((prot_len-2,3), name='ss_sparse_input')
        ensembled_seq = Concatenate()([model_aa, input_ss])
        model_out = self.DNN_body(ensembled_seq)
        model = Model(inputs=[input_aa, input_ss], outputs=[model_out])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )
        return model

    def fit(self, X_aa3mer_train, X_ss_sparse_train, y_train):
        if self.val_fold:
            skf = StratifiedKFold(n_splits=self.val_fold, shuffle=True)
            train, val = list(skf.split(X_aa3mer_train, y_train))[0]
            
            X_aa3mer_val = X_aa3mer_train[val, :self.maxlen-2]
            X_ss_sparse_val = X_ss_sparse_train[val, :self.maxlen-2]
            y_val = y_train[val]
            
            X_aa3mer_train = X_aa3mer_train[train, :self.maxlen-2]
            X_ss_sparse_train = X_ss_sparse_train[train, :self.maxlen-2]
            y_train = y_train[train]
            
            history = self.model.fit(
                [X_aa3mer_train, X_ss_sparse_train],
                to_categorical(y_train),
                batch_size=self.batch_size,
                epochs=200,
                validation_data=([X_aa3mer_val, X_ss_sparse_val], to_categorical(y_val)),
                class_weight=self.class_weight
            )
        else:
            X_aa3mer_train = X_aa3mer_train[:, :self.maxlen-2]
            X_ss_sparse_train = X_ss_sparse_train[:, :self.maxlen-2]
            
            history = self.model.fit(
                [X_aa3mer_train, X_ss_sparse_train],
                to_categorical(y_train),
                batch_size=self.batch_size,
                epochs=200,
                class_weight=self.class_weight
            )

        # Copy weights to long input models
        for model in self.model_long_input.values():
            model.set_weights(self.model.get_weights())
            
        return history

    def predict_score(self, X_aa3mer_test, X_ss_sparse_test, X_seqlens_test):
        scores = []
        for i in range(len(X_aa3mer_test)):
            if X_seqlens_test[i] <= self.maxlen:
                score = self.model.predict(
                    [
                        np.array([X_aa3mer_test[i][:self.maxlen-2]]),
                        np.array([X_ss_sparse_test[i][:self.maxlen-2]])
                    ],
                    verbose=0
                )[:,1][0]
            else:
                score = self.predict_long_seq(
                    X_aa3mer_test[i],
                    X_ss_sparse_test[i],
                    X_seqlens_test[i]
                )
            scores.append(score)
        return np.array(scores)

    def predict_long_seq(self, x_aa3mer, x_ss_sparse, x_seqlen):
        input_size = int(int(x_seqlen/(self.maxlen/2))*int(self.maxlen/2)+int(self.maxlen/2))
        input_size2 = min(self.max_seq_len, input_size)
        model = self.model_long_input[input_size2]
        score = model.predict(
            [
                np.array([x_aa3mer[:input_size2-2]]),
                np.array([x_ss_sparse[:input_size2-2]])
            ],
            verbose=0
        )[:,1][0]
        return score

    def save_model(self, filepath, name):
        self.model.save(os.path.join(filepath, f'{name}_model_param.h5'))

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        # Update long input models
        for model in self.model_long_input.values():
            model.set_weights(self.model.get_weights())

    def save_model2(self, filepath, name):
        # Save model architecture
        model_json = self.model.to_json()
        with open(os.path.join(filepath, f'{name}_model_structure.json'), 'w') as f:
            f.write(model_json)
        # Save weights
        self.model.save_weights(os.path.join(filepath, f'{name}_model_weights.h5'))

    def load_model2(self, json_file, weights_file):
        # Load model from JSON
        #with open(json_file, 'r') as f:
        #    model_json = f.read()
        model_json = json_file
        self.model = tf.keras.models.model_from_json(model_json)
        # Load weights
        self.model.load_weights(weights_file)
        # Update long input models
        for model in self.model_long_input.values():
            model.set_weights(self.model.get_weights())

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
    def __init__(self, BioVec_weights_add_null, max_seq_len=None, CNN_trainable=True,
                 class_weight={0:1., 1:9.}, dropout=0.3, maxlen=1500, batch_size=50,
                 val_fold=10, sliding_step=100, optimizer='Adam', n_gpus=1,
                 CPU='/device:CPU:0', autoEncoder_CNN=None, autoEncoder_Dense=None):
        """
        Difference from previous version: Also generate input-size-customized model for short proteins.
        Difference from previous version: use customized input size when doing prediction.
        """
        self.class_weight = class_weight
        self.maxlen = int(maxlen)
        self.max_seq_len = max_seq_len
        if self.max_seq_len:
            self.max_seq_len = int(self.max_seq_len)
        self.dropout = dropout
        self.val_fold = val_fold
        self.CNN_trainable = CNN_trainable
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.BioVec_weights_add_null = BioVec_weights_add_null
        self.autoEncoder_CNN = autoEncoder_CNN
        self.autoEncoder_Dense = autoEncoder_Dense
        self.sliding_step = sliding_step
        self.n_gpus = n_gpus
        self.CPU = CPU
        self.strategy = tf.distribute.MirroredStrategy() if n_gpus > 1 else None
        self.model = self.get_model()

        # Set up models for long input
        self.model_long_input = {}
        if self.max_seq_len and self.max_seq_len > self.maxlen:
            input_length = int(self.maxlen + self.maxlen/2)
            while input_length <= (self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+(self.maxlen/2):
                self.model_long_input[min(input_length, self.max_seq_len)] = \
                    self.get_model_customized_inputSize(min(input_length, self.max_seq_len))
                input_length += int(self.maxlen/2)

        # Set up models for short input
        self.model_short_input = {}
        for i in range(100, int(self.maxlen/100+1)*100+1, 100):
            self.model_short_input[int(i)] = self.get_model_customized_inputSize(int(i))

    def DNN_body(self, input_layer):
        # First Conv1D block
        if self.autoEncoder_CNN:
            weights = self.autoEncoder_CNN.get_layer(name='conv1').get_weights()
            ensembled_seq = Conv1D(
                filters=64, kernel_size=5, weights=weights,
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                padding='same', name='conv1', trainable=self.CNN_trainable
            )(input_layer)
        else:
            ensembled_seq = Conv1D(
                filters=64, kernel_size=5,
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                padding='same', name='conv1', trainable=self.CNN_trainable
            )(input_layer)

        ensembled_seq = BatchNormalization()(ensembled_seq)
        ensembled_seq = Activation('relu')(ensembled_seq)
        ensembled_seq = MaxPooling1D(pool_size=2, padding='same', name='maxpool1')(ensembled_seq)
        ensembled_seq = Dropout(self.dropout)(ensembled_seq)

        # Second Conv1D block
        if self.autoEncoder_CNN:
            weights = self.autoEncoder_CNN.get_layer(name='conv2').get_weights()
            ensembled_seq = Conv1D(
                filters=64, kernel_size=5, weights=weights,
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                padding='same', name='conv2', trainable=self.CNN_trainable
            )(ensembled_seq)
        else:
            ensembled_seq = Conv1D(
                filters=64, kernel_size=5,
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                padding='same', name='conv2', trainable=self.CNN_trainable
            )(ensembled_seq)

        ensembled_seq = BatchNormalization()(ensembled_seq)
        ensembled_seq = Activation('relu')(ensembled_seq)
        ensembled_seq = MaxPooling1D(pool_size=2, padding='same', name='maxpool2')(ensembled_seq)
        ensembled_seq = Dropout(self.dropout)(ensembled_seq)

        # Global pooling and dense layers
        ensembled_seq = GlobalMaxPooling1D()(ensembled_seq)
        ensembled_seq = Dropout(self.dropout)(ensembled_seq)

        # Dense layers
        if self.autoEncoder_Dense:
            model = Dense(
                21, weights=self.autoEncoder_Dense.get_layer(name='dense1').get_weights(),
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense1'
            )(ensembled_seq)
        else:
            model = Dense(
                21, kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01), name='dense1'
            )(ensembled_seq)

        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(self.dropout)(model)

        if self.autoEncoder_Dense:
            model = Dense(
                5, weights=self.autoEncoder_Dense.get_layer(name='dense2').get_weights(),
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense2'
            )(model)
        else:
            model = Dense(
                5, kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01), name='dense2'
            )(model)

        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(self.dropout)(model)

        model_out = Dense(2, activation='sigmoid')(model)
        return model_out

    def get_model(self):
        if self.n_gpus > 1:
            with self.strategy.scope():
                return self._build_model()
        else:
            return self._build_model()

    def _build_model(self):
        input_aa = Input((self.maxlen-2,), name='aa3mer_input')
        model_aa = Embedding(
            input_dim=self.BioVec_weights_add_null.shape[0],
            output_dim=100,
            input_length=self.maxlen-2,
            weights=[self.BioVec_weights_add_null],
            trainable=False,
            name='aa3mer_embedding'
        )(input_aa)

        input_ss = Input((self.maxlen-2,3), name='ss_sparse_input')
        ensembled_seq = Concatenate()([model_aa, input_ss])
        model_out = self.DNN_body(ensembled_seq)
        model = Model(inputs=[input_aa, input_ss], outputs=[model_out])

        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )
        return model

    def get_model_customized_inputSize(self, prot_len):
        prot_len = int(prot_len)
        if self.n_gpus > 1:
            with self.strategy.scope():
                return self._build_model_customized(prot_len)
        else:
            return self._build_model_customized(prot_len)

    def _build_model_customized(self, prot_len):
        input_aa = Input((prot_len-2,), name='aa3mer_input')
        model_aa = Embedding(
            input_dim=self.BioVec_weights_add_null.shape[0],
            output_dim=100,
            input_length=prot_len-2,
            weights=[self.BioVec_weights_add_null],
            trainable=False,
            name='aa3mer_embedding'
        )(input_aa)

        input_ss = Input((prot_len-2,3), name='ss_sparse_input')
        ensembled_seq = Concatenate()([model_aa, input_ss])
        model_out = self.DNN_body(ensembled_seq)
        model = Model(inputs=[input_aa, input_ss], outputs=[model_out])

        model.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )
        return model

    def fit(self, X_aa3mer_train, X_ss_sparse_train, y_train):
        if self.val_fold:
            skf = StratifiedKFold(n_splits=self.val_fold, shuffle=True)
            train, val = list(skf.split(X_aa3mer_train, y_train))[0]

            X_aa3mer_val = X_aa3mer_train[val, :self.maxlen-2]
            X_ss_sparse_val = X_ss_sparse_train[val, :self.maxlen-2]
            y_val = y_train[val]

            X_aa3mer_train = X_aa3mer_train[train, :self.maxlen-2]
            X_ss_sparse_train = X_ss_sparse_train[train, :self.maxlen-2]
            y_train = y_train[train]

            history = self.model.fit(
                [X_aa3mer_train, X_ss_sparse_train],
                to_categorical(y_train),
                batch_size=self.batch_size,
                epochs=200,
                validation_data=([X_aa3mer_val, X_ss_sparse_val], to_categorical(y_val)),
                class_weight=self.class_weight
            )
        else:
            X_aa3mer_train = X_aa3mer_train[:, :self.maxlen-2]
            X_ss_sparse_train = X_ss_sparse_train[:, :self.maxlen-2]

            history = self.model.fit(
                [X_aa3mer_train, X_ss_sparse_train],
                to_categorical(y_train),
                batch_size=self.batch_size,
                epochs=200,
                class_weight=self.class_weight
            )

        # Copy weights to both long and short input models
        for model in self.model_long_input.values():
            model.set_weights(self.model.get_weights())
        for model in self.model_short_input.values():
            model.set_weights(self.model.get_weights())

        return history

    def predict_score(self, X_aa3mer_test, X_ss_sparse_test, X_seqlens_test):
        scores = []
        for i in range(len(X_aa3mer_test)):
            if X_seqlens_test[i] <= self.maxlen/2:
                score = self.predict_short_seq(
                    X_aa3mer_test[i],
                    X_ss_sparse_test[i],
                    X_seqlens_test[i]
                )
            elif X_seqlens_test[i] <= self.maxlen:
                score = self.model.predict(
                    [
                        np.array([X_aa3mer_test[i][:self.maxlen-2]]),
                        np.array([X_ss_sparse_test[i][:self.maxlen-2]])
                    ],
                    verbose=0
                )[:,1][0]
            else:
                score = self.predict_long_seq(
                    X_aa3mer_test[i],
                    X_ss_sparse_test[i],
                    X_seqlens_test[i]
                )
            scores.append(score)
        return np.array(scores)

    def predict_long_seq(self, x_aa3mer, x_ss_sparse, x_seqlen):
        input_size = int(int(x_seqlen/(self.maxlen/2))*int(self.maxlen/2)+int(self.maxlen/2))
        model = self.model_long_input[min(self.max_seq_len, input_size)]
        score = model.predict(
            [
                np.array([x_aa3mer[:input_size-2]]),
                np.array([x_ss_sparse[:input_size-2]])
            ],
            verbose=0
        )[:,1][0]
        return score

    def predict_short_seq(self, x_aa3mer, x_ss_sparse, x_seqlen):
        input_size = int(int(x_seqlen-1)/100+1)*100
        model = self.model_short_input[min(self.max_seq_len, input_size)]
        score = model.predict(
            [
                np.array([x_aa3mer[:input_size-2]]),
                np.array([x_ss_sparse[:input_size-2]])
            ],
            verbose=0
        )[:,1][0]
        return score

    def save_model(self, filepath, name):
        self.model.save(os.path.join(filepath, f'{name}_model_param.h5'))

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        # Update both long and short input models
        for model in self.model_long_input.values():
            model.set_weights(self.model.get_weights())
        for model in self.model_short_input.values():
            model.set_weights(self.model.get_weights())

    def save_model2(self, filepath, name):
        # Save model architecture
        model_json = self.model.to_json()
        with open(os.path.join(filepath, f'{name}_model_structure.json'), 'w') as f:
            f.write(model_json)
        # Save weights
        self.model.save_weights(os.path.join(filepath, f'{name}_model_weights.h5'))

    def load_model2(self, json_file, weights_file):
        # Load model from JSON
        #with open(json_file, 'r') as f:
        #    model_json = f.read()
        model_json = json_file
        self.model = tf.keras.models.model_from_json(model_json)
        # Load weights
        self.model.load_weights(weights_file)
        # Update long input models
        for model in self.model_long_input.values():
            model.set_weights(self.model.get_weights())

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

def autoEncoder_training(
    out_dir,
    BioVec_weights_add_null,
    X_aa3mer,
    X_ss_sparse=None,
    noSS=False,
    model_name='Model',
    maxlen=1500,
    batch_size=128,
    epochs=300,
    optimizer='nadam',
    loss='mse'
):
    """
    Train autoencoder models for protein sequence data.
    
    Args:
        out_dir: Output directory for saving models
        BioVec_weights_add_null: Embedding weights matrix
        X_aa3mer: Amino acid sequence data
        X_ss_sparse: Secondary structure data (optional)
        noSS: Boolean to indicate if secondary structure should be ignored
        model_name: Name prefix for saved models
        maxlen: Maximum sequence length
        batch_size: Batch size for training
        epochs: Number of training epochs
        optimizer: Optimizer name or instance
        loss: Loss function name or instance
    """
    if (X_ss_sparse is None) and (not noSS):
        raise ValueError('Secondary structure data should be provided.')

    # Prepare the input data
    input_aa = Input((maxlen,), name='aa3mer_input')
    model_aa = Embedding(
        input_dim=BioVec_weights_add_null.shape[0],
        output_dim=100,
        input_length=maxlen,
        weights=[BioVec_weights_add_null * 10],
        trainable=False,
        name='aa3mer_embedding'
    )(input_aa)

    if noSS:
        prepare = Model(inputs=[input_aa], outputs=[model_aa])
        prepare.compile(
            loss='binary_crossentropy',
            optimizer='Adam',
            metrics=['accuracy']
        )
        emsembled_train = prepare.predict([X_aa3mer], batch_size=1, verbose=0)
    else:
        input_ss = Input((maxlen, 3), name='ss3mer_input')
        ensembled_seq = Concatenate()([model_aa, input_ss])
        prepare = Model(inputs=[input_aa, input_ss], outputs=[ensembled_seq])
        prepare.compile(
            loss='binary_crossentropy',
            optimizer='Adam',
            metrics=['accuracy']
        )
        emsembled_train = prepare.predict(
            [X_aa3mer, X_ss_sparse],
            batch_size=1,
            verbose=0
        )

    # Define CNN Autoencoder
    input_dim = 100 if noSS else 103
    input_seq = Input(shape=(maxlen, input_dim))
    
    # Common parameters for Conv1D layers
    conv_params = {
        'kernel_size': 5,
        'kernel_initializer': 'glorot_uniform',
        'bias_initializer': 'zeros',
        'kernel_regularizer': regularizers.l2(0.01),
        'bias_regularizer': regularizers.l2(0.01),
        'padding': 'same',
        'trainable': True
    }

    # Encoder
    ensembled_seq = Conv1D(filters=64, name='conv1', **conv_params)(input_seq)
    ensembled_seq = BatchNormalization()(ensembled_seq)
    ensembled_seq = Activation('relu')(ensembled_seq)
    ensembled_seq = MaxPooling1D(pool_size=2, padding='same', name='maxpool1')(ensembled_seq)
    
    ensembled_seq = Conv1D(filters=64, name='conv2', **conv_params)(ensembled_seq)
    ensembled_seq = BatchNormalization()(ensembled_seq)
    ensembled_seq = Activation('relu')(ensembled_seq)
    encoded = MaxPooling1D(pool_size=2, padding='same', name='maxpool2')(ensembled_seq)

    # Decoder
    ensembled_seq = Conv1D(filters=64, name='conv3', **conv_params)(encoded)
    ensembled_seq = BatchNormalization()(ensembled_seq)
    ensembled_seq = Activation('relu')(ensembled_seq)
    ensembled_seq = UpSampling1D(size=2, name='upsampling1')(ensembled_seq)
    
    ensembled_seq = Conv1D(filters=64, name='conv4', **conv_params)(ensembled_seq)
    ensembled_seq = BatchNormalization()(ensembled_seq)
    ensembled_seq = Activation('relu')(ensembled_seq)
    ensembled_seq = UpSampling1D(size=2, name='upsampling2')(ensembled_seq)
    
    # Final layer
    decoded = Conv1D(
        filters=input_dim,
        activation='tanh',
        name='conv5',
        **conv_params
    )(ensembled_seq)

    # Create and compile CNN autoencoder
    autoencoder_CNN = Model(inputs=[input_seq], outputs=[decoded])
    autoencoder_CNN.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Train CNN autoencoder
    history = autoencoder_CNN.fit(
        emsembled_train,
        emsembled_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True
    )

    # Save CNN autoencoder
    autoencoder_CNN.save(
        os.path.join(out_dir, f'{model_name}_CNN_autoencoder_withRBPs.h5')
    )

    # Create and train Dense autoencoder
    gbmax_encoded = GlobalMaxPooling1D()(encoded)
    model_dense = Model(inputs=[input_seq], outputs=[gbmax_encoded])
    model_dense.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )
    CNN_out_train = model_dense.predict(emsembled_train, verbose=0)

    # Dense autoencoder architecture
    dense_params = {
        'kernel_initializer': 'glorot_uniform',
        'bias_initializer': 'zeros',
        'kernel_regularizer': regularizers.l2(0.01),
        'bias_regularizer': regularizers.l2(0.01),
        'trainable': True
    }

    dense_input = Input((64,), name='DenseI_input')
    
    # Encoder
    encoded_dense = Dense(21, activation='relu', name='dense1', **dense_params)(dense_input)
    encoded_dense = Dense(5, activation='relu', name='dense2', **dense_params)(encoded_dense)
    
    # Decoder
    decoded_dense = Dense(21, activation='relu', name='dense3', **dense_params)(encoded_dense)
    decoded_dense = Dense(64, activation='tanh', name='dense4', **dense_params)(decoded_dense)

    # Create and compile Dense autoencoder
    autoencoder_dense = Model(inputs=[dense_input], outputs=[decoded_dense])
    autoencoder_dense.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )

    # Train Dense autoencoder
    history = autoencoder_dense.fit(
        CNN_out_train,
        CNN_out_train,
        epochs=epochs,
        batch_size=64,
        shuffle=True
    )

    # Save Dense autoencoder
    autoencoder_dense.save(
        os.path.join(out_dir, f'{model_name}_DenseLayer_autoencoder_withRBPs.h5')
    )
    
    return autoencoder_CNN, autoencoder_dense
