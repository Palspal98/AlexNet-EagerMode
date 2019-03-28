import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.keras.layers as tkl

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPool2D,BatchNormalization

import numpy as np
from collections import OrderedDict

class AlexNet(tf.keras.Model):
	def __init__(self,classes):
		super(AlexNet, self).__init__()
		self.p=1
		self.conv1 = Sequential()
		self.conv1.add(Conv2D(filters=96, kernel_size=(11,11),strides=(4,4), padding='valid'))
		
		self.conv2 = Sequential()		
		self.conv2.add(Activation('relu'))
		self.conv2.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
		self.conv2.add(BatchNormalization())
		self.conv2.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))

		self.conv3 = Sequential()		
		self.conv3.add(Activation('relu'))
		self.conv3.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
		self.conv3.add(BatchNormalization())
		self.conv3.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))


		self.conv4 = Sequential()		
		self.conv4.add(Activation('relu'))
		self.conv4.add(BatchNormalization())
		self.conv4.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))

		self.conv5 = Sequential()
		self.conv5.add(Activation('relu'))
		self.conv5.add(BatchNormalization())
		self.conv5.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))

		self.get_fc_val = Sequential()
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))
		self.get_fc_val.add(Flatten())
		self.get_fc_val.add(Dense(4096))
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(Dropout(0.4))
		self.get_fc_val.add(Dense(4096))
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(Dropout(0.4))
		self.get_cl_val = Sequential()
		self.get_cl_val.add(Dense(classes))

	def call(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)
		conv5 = self.conv5(conv4)
		if self.p:
			fc_val = self.get_fc_val(conv5)
			return self.get_cl_val(fc_val)
		else:
			return conv1,conv2,conv3,conv4,conv5

class GAP(tf.keras.Model):
	"""docstring for GAP"""
	def __init__(self, ):
		super(GAP, self).__init__()
		self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
	def call(self,x):
		return self.global_pool(x)
		

class AlexNetGAP(tf.keras.Model):
	def __init__(self,classes):
		super(AlexNetGAP, self).__init__()
		self.p=1
		self.conv1 = Sequential()
		self.conv1.add(Conv2D(filters=96, kernel_size=(11,11),strides=(4,4), padding='valid'))
		
		self.conv2 = Sequential()		
		self.conv2.add(Activation('relu'))
		self.conv2.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
		self.conv2.add(BatchNormalization())
		self.conv2.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))

		self.conv3 = Sequential()		
		self.conv3.add(Activation('relu'))
		self.conv3.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
		self.conv3.add(BatchNormalization())
		self.conv3.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))


		self.conv4 = Sequential()		
		self.conv4.add(Activation('relu'))
		self.conv4.add(BatchNormalization())
		self.conv4.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))

		self.conv5 = Sequential()
		self.conv5.add(Activation('relu'))
		self.conv5.add(BatchNormalization())
		self.conv5.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
		self.conv5.add(Activation('relu'))
		self.global_pool = GAP()
		self.get_fc_val = Sequential()
		self.get_fc_val.add(Dropout(0.4))
		self.get_fc_val.add(Dense(4096))
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(Dropout(0.4))
		self.get_fc_val.add(Dense(4096))
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(Dropout(0.4))
		self.get_cl_val = Sequential()
		self.get_cl_val.add(Dense(classes))

	def call(self, x):
		conv1 = self.conv1(x)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)
		conv5 = self.conv5(conv4)
		conv = self.global_pool(conv5)
		if self.p:
			fc_val = self.get_fc_val(conv)
			return self.get_cl_val(fc_val)
		else:
			conv = tf.expand_dims(tf.expand_dims(tf.nn.softmax(conv,axis=-1),1),1)
			return tf.reduce_sum((conv5*conv),axis=-1)
class AttentionEstimator(tf.keras.Model):
	def __init__(self,inputs):
		super(AttentionEstimator,self).__init__()

		self.get_fc_val = Sequential()
		self.get_fc_val.add(Dense(1))

	def call(self,conv,fc):
		b,w,h,c = conv.shape
		spatial_conv = tf.reshape(conv,[b,w*h,c])
		f_expanded = tf.expand_dims(fc,1)
		c_fc = tf.tile(f_expanded,[1,w*h,1])
		vector = tf.reshape(tf.concat([c_fc,spatial_conv],axis=2),[b*w*h,-1])
		attention = tf.keras.activations.softmax(tf.reshape(self.get_fc_val(vector),[b,w*h]))
		attention_e = tf.expand_dims(attention,-1)
		return tf.reduce_sum(spatial_conv*attention_e,axis=1),attention_e


class AttentionSpatialEstimator(tf.keras.Model):
	def __init__(self,inputs):
		super(AttentionSpatialEstimator,self).__init__()

		self.get_fc_val = Sequential()
		self.get_fc_val.add(Dense(inputs))

	def call(self,conv,fc):
		b,w,h,c = conv.shape
		spatial_conv = tf.reshape(conv,[b,w*h,c])
		C_attn = tf.expand_dims(self.get_fc_val(fc),1)
		C = tf.keras.activations.softmax(tf.reduce_mean(spatial_conv*C_attn,axis=2))
		spatial_map = tf.reduce_sum(spatial_conv*tf.expand_dims(C,-1),axis=1)
		return spatial_map,C


class AlexAttNet(tf.keras.Model):
	def __init__(self,classes):
		super(AlexAttNet, self).__init__()
		self.p=1
		self.till_conv4 = Sequential()
		self.till_conv4.add(Conv2D(filters=96, kernel_size=(11,11),strides=(4,4), padding='valid'))
		self.till_conv4.add(Activation('relu'))
		self.till_conv4.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
		self.till_conv4.add(BatchNormalization())
		self.till_conv4.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
		self.till_conv4.add(Activation('relu'))
		self.till_conv4.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
		self.till_conv4.add(BatchNormalization())

		self.till_conv4.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
		self.till_conv4.add(Activation('relu'))
		self.till_conv4.add(BatchNormalization())

		self.till_conv4.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))

		self.till_conv5 = Sequential()
		self.till_conv5.add(Activation('relu'))
		self.till_conv5.add(BatchNormalization())
		self.till_conv5.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))

		self.get_fc_val = Sequential()
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))
		self.get_fc_val.add(Flatten())
		self.get_fc_val.add(Dense(4096))
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(Dropout(0.4))
		self.get_fc_val.add(Dense(4096))
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(Dropout(0.4))
		
		self.conv4_Att = AttentionEstimator(1)
		self.flatten_conv4 = Flatten()
		
		self.conv5_Att = AttentionEstimator(1)
		self.flatten_conv5 = Flatten()

		self.get_cl_val = Sequential()
		self.get_cl_val.add(Dense(classes))


		
	def call(self, x):
		conv4 = self.till_conv4(x)
		conv5 = self.till_conv5(conv4)
		fcs = self.get_fc_val(conv5)
		if self.p==1:
			conv4_att = self.flatten_conv4(self.conv4_Att(conv4,fcs)[0])
			conv5_att = self.flatten_conv5(self.conv5_Att(conv5,fcs)[0])
			f_fcs = tf.concat(axis=-1,values=[conv4_att,conv5_att])
			out = self.get_cl_val(f_fcs)
			return out
		else:
			conv4_att = self.conv4_Att(conv4,fcs)[1]
			conv5_att = self.conv5_Att(conv5,fcs)[1]
			return conv4_att,conv5_att




class AlexNetAttSpatial(tf.keras.Model):
	def __init__(self,classes):
		super(AlexNetAttSpatial, self).__init__()
		self.p=1
		self.till_conv4 = Sequential()
		self.till_conv4.add(Conv2D(filters=96, kernel_size=(11,11),strides=(4,4), padding='valid'))
		self.till_conv4.add(Activation('relu'))
		self.till_conv4.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
		self.till_conv4.add(BatchNormalization())
		self.till_conv4.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
		self.till_conv4.add(Activation('relu'))
		self.till_conv4.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
		self.till_conv4.add(BatchNormalization())

		self.till_conv4.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
		self.till_conv4.add(Activation('relu'))
		self.till_conv4.add(BatchNormalization())

		self.till_conv4.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))

		self.till_conv5 = Sequential()
		self.till_conv5.add(Activation('relu'))
		self.till_conv5.add(BatchNormalization())

		self.till_conv5.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))

		self.get_fc_val = Sequential()
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))
		self.get_fc_val.add(Flatten())
		self.get_fc_val.add(Dense(4096))
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(Dropout(0.4))
		self.get_fc_val.add(Dense(4096))
		self.get_fc_val.add(Activation('relu'))
		self.get_fc_val.add(Dropout(0.4))

		self.conv4_Att = AttentionSpatialEstimator(384)
		self.flatten_conv4 = Flatten()
		
		self.conv5_Att = AttentionSpatialEstimator(256)
		self.flatten_conv5 = Flatten()

		self.get_cl_val = Sequential()
		self.get_cl_val.add(Dense(classes))


		
	def call(self, x):
		conv4 = self.till_conv4(x)
		conv5 = self.till_conv5(conv4)
		fcs = self.get_fc_val(conv5)
		if self.p==1:
			conv4_att = self.flatten_conv4(self.conv4_Att(conv4,fcs)[0])
			conv5_att = self.flatten_conv5(self.conv5_Att(conv5,fcs)[0])
			f_fcs = tf.concat(axis=-1,values=[conv4_att,conv5_att])
			out = self.get_cl_val(f_fcs)
			return out
		else:
			conv4_att = self.conv4_Att(conv4,fcs)[1]
			conv5_att = self.conv5_Att(conv5,fcs)[1]
			return conv4_att,conv5_att
