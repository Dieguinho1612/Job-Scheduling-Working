#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Concatenate, LeakyReLU, Softmax, Dropout
from keras.layers import Lambda, Flatten, Bidirectional, TimeDistributed, Reshape, MultiHeadAttention, LayerNormalization
from keras.activations import tanh
from keras.models import Model, Sequential
import keras.backend
import random
from copy import copy


# In[ ]:


def costs(y_true, y_pred):
    
    indices = keras.backend.argmax(y_pred, axis=1)
    length = np.shape(indices)[0]
    inv_values = keras.backend.eval(y_true)[np.arange(length),indices]
    Q_factors = 1/inv_values
    Q_factor = np.mean(Q_factors)
    
    Q_factor = (Q_factor - 1)*100
    Q_factor = round(Q_factor,2) #vllt mal 10.000 nehmen, int verwandeln und durch 100 teilen
    
    return Q_factor


# In[ ]:


def MSE_with_Softmax(y_pred, y_true):
    y_pred = Softmax()(y_pred)
    y_true = Softmax()(y_true)
    loss = keras.losses.MeanSquaredError()(y_pred,y_true)
    #return loss
    return (loss * 10e4)


# In[2]:


class FeedForward(keras.layers.Layer):
    def __init__(self, inner_dim, outer_dim, alpha=0.1, name=None,**kwargs):
        super(FeedForward, self).__init__(name=name,**kwargs)
        self.outer_dim = outer_dim
        self.inner_dim = inner_dim
        self.alpha = alpha
        self.dense_in = Dense(self.inner_dim)
        self.dense_out = Dense(self.outer_dim)
        
    def call(self, x):
        x = self.dense_in(x)
        x = LeakyReLU(self.alpha)(x)
        x = self.dense_out(x)
        return x
   
    def get_config(self):
        config = super().get_config()
        config.update({
            "outer_dim": self.outer_dim,
            "inner_dim": self.inner_dim,
            "alpha": self.alpha
        })
        return config


# In[ ]:


class Pointer(keras.layers.Layer):
    def __init__(self, dim, name=None,**kwargs):
        super(Pointer, self).__init__(name=name,**kwargs)
        self.dim = dim
        self.W1 = Dense(dim, use_bias=False)
        self.W2 = Dense(dim, use_bias=False)
        self.V = Dense(1, use_bias=False)
        
    def call(self, enc_outputs, dec_output):
        w1_e = self.W1(enc_outputs)
        w2_d = Reshape((1,-1))(self.W2(dec_output))
        tanh_out = tanh(w1_e + w2_d)
        attention = self.V(tanh_out)
        out = Flatten()(attention)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim
        })
        return config


# In[ ]:


class TransformerLayer(keras.layers.Layer):
    def __init__(self,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
                 self_att,
                 alpha=0.1, # Leaky-ReLU Parameter
               dropout_rate=0, #0.1,
                 name=None,**kwargs
               ):
        super(TransformerLayer,self).__init__(name=name,**kwargs)
        
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.dff = dff
        self.self_att = self_att
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        
        # Multi-head self-attention.
        self.mha = MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model, # Size of each attention head for query Q and key K.
            dropout=dropout_rate,
            )
        # Point-wise feed-forward network.
        self.ff = FeedForward(d_model, dff, alpha=alpha)

        # Layer normalization.
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = Dropout(dropout_rate)

    def call(self, x, training=True):

        if self.self_att==True:
            query = x #self attention
            val = x
        else:
            query = x[:,0:1,:]
            #query = Reshape((1,-1))(query)
            val = x[:,1:,:]
            
        
        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output = self.mha(
            query=query,  # Query Q tensor.
            value=val,  # Value V tensor.
            key=val,  # Key K tensor.
            training=training, # A boolean indicating whether the layer should behave in training mode.
            )

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(query + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`

        # Point-wise feed-forward network output.
        ff_output = self.ff(out1)  # Shape `(batch_size, input_seq_len, d_model)`
        ff_output = self.dropout1(ff_output, training=training)
        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layernorm2(out1 + ff_output)  # Shape `(batch_size, input_seq_len, d_model)`.

        if self.self_att:
            return out2
        else:
            #output is row in matrix shape of 1,dimension. Get rid of that "1"
            out2 = Flatten()(out2)
            return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_attention_heads": self.num_attention_heads,
            "dff": self.dff,
            "self_att": self.self_att,
            "alpha": self.alpha,
            "dropout_rate": self.dropout_rate
        })
        return config
        
    
    def compute_output_shape(self, input_shape):
        if self.self_att==True:
            return (input_shape)
        else:
            shp = input_shape
            return ((shp[0],shp[2]))


# In[ ]:


"""Merge the Machine/Runtime Info with the Jobs Deadline/Weights Info to pass it to the Transformer Decoder Attention Part"""
class Merge(keras.layers.Layer):
    def __init__(self, name=None,**kwargs):
        super(Merge, self).__init__(name=name,**kwargs)
        
    def call(self, x_j, x_m):
        x_j = TimeDistributed(Reshape((1,-1)))(x_j)
        x = Concatenate(axis=-2)([x_j,x_m])
        return x


# In[ ]:


"""Add Zero Line at the Bottom"""
class AddZeroLine(keras.layers.Layer):
    def __init__(self, name=None,**kwargs):
        super(AddZeroLine, self).__init__(name=name,**kwargs)
        
    def call(self, x):
        zero_line = tf.zeros_like(x[:,0:1,:])
        x = Concatenate(axis=1)([x,zero_line])
        return x


# In[1]:


"""Evaluation Functions and Model Build:"""


# In[ ]:


def costs(y_true, y_pred):
    
    indices = keras.backend.argmax(y_pred, axis=1)
    length = np.shape(indices)[0]
    inv_values = keras.backend.eval(y_true)[np.arange(length),indices]
    Q_factors = 1/inv_values
    Q_factor = np.mean(Q_factors)
    
    Q_factor = (Q_factor - 1)*100
    Q_factor = round(Q_factor,2) #vllt mal 10.000 nehmen, int verwandeln und durch 100 teilen
    
    return Q_factor


# In[ ]:


def MSE_with_Softmax(y_pred, y_true):
    y_pred = Softmax()(y_pred)
    y_true = Softmax()(y_true)
    loss = keras.losses.MeanSquaredError()(y_pred,y_true)
    #return loss
    return (loss * 10e3)


# In[ ]:


def build_Transformer():
   
   """Dropout muss noch hinzugefügt werden"""
   inputs_machines = keras.Input(shape=(None,None,4), name="Machine_Info_with_Runtimes")
   inputs_jobs = keras.Input(shape=(None,2), name="Jobs_Deadlines/Weights")

   x_m = TimeDistributed(Bidirectional(LSTM(16)))(inputs_machines)
   x_m = FeedForward(16,16)(x_m)

   x_j = FeedForward(4,4)(inputs_jobs)
   x_j = MultiHeadAttention(num_heads=2, key_dim=8)(x_j, x_j)
   x_j = FeedForward(4,4)(x_j)

   x = Concatenate()([x_m,x_j])
   x = FeedForward(16,16)(x)

   x = MultiHeadAttention(num_heads=4, key_dim=32)(x,x)
   x = FeedForward(16,16)(x)

   x_enc, *enc_state = Bidirectional(LSTM(32, return_sequences=True, return_state=True, name="Encoder"))(x)
   #x_enc = MultiHeadAttention(num_heads=4, key_dim=64)(x_enc,x_enc)
   #x_enc = FeedForward(64,64)(x_enc)

   x_dec = Bidirectional(LSTM(32, name="Decoder"))(x_enc, initial_state=enc_state)

   shtdwn = FeedForward(1,32)(x_dec) 
   val = FeedForward(32,32)(x_enc)
   query = FeedForward(32,32)(x_dec)

   outputs = Pointer(64)(val,query)
   output = Concatenate()([outputs,shtdwn])



   #query_shtdwn = FeedForward(32,64)(x_dec)
   #query_shtdwn = Reshape((1,-1))(query_shtdwn)
   #val_shtdwn = FeedForward(32,64)(x_enc)
   #x_shtdwn = MultiHeadAttention(num_heads=8, key_dim=16)(query=query_shtdwn, value=val_shtdwn)
   #x_shtdwn = Flatten()(x_shtdwn)
   #x_shtdwn = FeedForward(1,8)(x_shtdwn)

   #output = Concatenate()([outputs,x_shtdwn])

   Pointer_with_Attention = keras.Model(inputs=[inputs_machines,inputs_jobs], outputs=output)
   
   return Pointer_with_Attention

