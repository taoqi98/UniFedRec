from hypers import *
import keras
from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten, average,Lambda

from keras.layers import *
from keras.models import Model, load_model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.optimizers import *

class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    user_vecs =Dropout(0.2)(vecs_input)
    user_att = Dense(200,activation='tanh')(user_vecs)
    user_att = keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model(vecs_input,user_vec)
    return model

def AttentivePoolingAMask(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    mask_input = Input(shape=(dim1,),dtype='float32')

    user_vecs =Dropout(0.2)(vecs_input)
    user_att = Dense(200,activation='tanh')(user_vecs)
    user_att = Dense(1)(user_att)
    user_att = keras.layers.Reshape((dim1,))(user_att)
    user_att = keras.layers.Lambda(lambda x:x[0]-1000*(1-x[1]))([user_att,mask_input])
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model([vecs_input,mask_input],user_vec)
    return model

def GlobalAvgPool1DAMask(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    mask_input = Input(shape=(dim1,),dtype='float32')
    
    s = keras.layers.Lambda(lambda x:K.sum(x,axis=-1,keepdims=True))(mask_input)
    mask = keras.layers.Lambda(lambda x:x[0]/(x[1]+10**(-6)))([mask_input,s])
    mask = keras.layers.Reshape((dim1,1))(mask)
    vecs = keras.layers.Lambda(lambda x:x[0]*x[1])([vecs_input,mask])
    vec = keras.layers.Lambda(lambda x: K.sum(x,axis=-2))(vecs)

    return Model([vecs_input,mask_input],vec)

class BaseLayerV2(Layer):
 
    def __init__(self, base_num, dim, **kwargs):
        self.base_num = base_num
        self.dim = dim
        super(BaseLayerV2, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.key_matrix = self.add_weight(name='Key',
                                  shape=(self.dim,self.base_num,),
                                  initializer='uniform',
                                  trainable=True)
        
        self.value_matrix = self.add_weight(name='Value',
                                  shape=(self.base_num,self.dim,),
                                  initializer='uniform',
                                  trainable=True)

        super(BaseLayerV2, self).build(input_shape)
 
    def call(self, user_vec):
        
        # user_vec (bz,400)
        
        att = K.dot(user_vec,self.key_matrix) # user_vec (bz,base_num)
        att = K.exp(att)
        weight = att/(K.sum(att,axis=-1,keepdims=True)+10**(-6)) #(bz,base_num)
        
#         value_matrix = K.permute_dimensions(self.key_matrix,(1,0))
#         vec = K.dot(weight,value_matrix)
        
        vec = K.dot(weight,self.value_matrix)
        
        return vec
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

def get_user_model():
    user_vecs_input = Input(shape=(50,256))
    mask_input = Input(shape=(MAX_CHANNEL_NUM,50,))
    
    user_vecs = Dropout(0.2)(user_vecs_input)
     
    user_vecs = Attention(16//2,16*2)([user_vecs]*3)
    att_layer = AttentivePoolingAMask(50,256)
    base_layer = BaseLayerV2(BaseNum,256)
    uvs = []
    for i in range(MAX_CHANNEL_NUM):
        mask = keras.layers.Lambda(lambda x:x[:,i,:])(mask_input)
        uv = att_layer([user_vecs,mask])
        uv = base_layer(uv)
        uv = keras.layers.Reshape((256,1))(uv)
        uvs.append(uv)
    uvs = keras.layers.Concatenate(axis=-1)(uvs)
        
    return Model([user_vecs_input,mask_input],uvs), Model(user_vecs_input,user_vecs),

def create_model(lr,clipvalue):
        
    user_model, user_vecs_model = get_user_model()
    
    clicked_title_input = Input(shape=(50,256,), dtype='float32')    
    clicked_mask = Input(shape=(MAX_CHANNEL_NUM,50,), dtype='float32')    
    quotas_input = Input(shape=(1+npratio,MAX_CHANNEL_NUM),dtype='float32')
    
    title_inputs = Input(shape=(1+npratio,256,),dtype='float32') 
    
    clicked_news_vecs = clicked_title_input
    title_vecs = title_inputs

    uvs = user_model([clicked_news_vecs,clicked_mask]) #(bz,dim,C)
    
    title_vecs = Dropout(0.2)(title_vecs) #(bz,1+npratio,dim)
    scores = keras.layers.Dot([-1,-2])([title_vecs,uvs]) #(bz,1+npratio,C)
    
    scores = keras.layers.Multiply()([quotas_input,scores])
    scores = keras.layers.Lambda(lambda x:K.sum(x,axis=-1))(scores)
    
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)     

    model = Model([title_inputs, clicked_title_input,clicked_mask,quotas_input],logits) # max prob_click_positive
    model.compile(loss=['categorical_crossentropy'],
                      optimizer= SGD(lr=lr,clipvalue=clipvalue),
                      metrics=['acc'])

    return model,user_model,user_vecs_model,