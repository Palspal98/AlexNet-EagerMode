from tensorflow.python.ops import lookup_ops
import tensorflow as tf
from collections import OrderedDict
import pickle
from scipy.misc import imresize
import scipy.io
import numpy as np
from sklearn.utils import shuffle

class cifar_10_dataset(object):
	"""docstring for cifar_10_dataset"""
	def __init__(self, filename,classes=10,batch_size=32,buffer_size=1024,norm_type='Constant',prefetch=1):
		super(cifar_10_dataset, self).__init__()
		self.filename = filename
		self.batch_size = batch_size
		self.buffer_size = buffer_size
		self.prefetch = prefetch
		self.fetch_data()
		self.classes = classes
		self.train_ds = self.datasetGenerator(self.train_img_data,self.train_lbl_data)
		self.test_ds = self.datasetGenerator(self.test_img_data,self.test_lbl_data)
		self.mode = 'train'

	def fetch_data(self):
		train_img_data = []
		train_lbl_data = []
		test_img_data = []
		test_lbl_data = []
		self.train_img_data = []
		self.test_img_data = []
		for i in range(5):
			with open(self.filename+'data_batch_%d'%(i+1), 'rb') as fo:
				dict = pickle.load(fo, encoding='bytes')
				train_img_data.append(dict[b'data'].reshape(-1,3,32,32))
				train_lbl_data.append(np.asarray(dict[b'labels']))
		
		self.train_img_data = np.transpose(np.vstack(train_img_data),(0,2,3,1))
		self.train_lbl_data = np.hstack(train_lbl_data)

		with open(self.filename+'test_batch', 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
			test_img_data = dict[b'data'].reshape(-1,3,32,32)
			self.test_lbl_data = np.asarray(dict[b'labels'])

		self.test_img_data = np.transpose(test_img_data,(0,2,3,1))
		

	
	def datasetGenerator(self,images,labels):
		def build_gen(img,lbl):
			return tf.image.resize_images(img,size=(227,227))/255.0,lbl
		# ds = tf.data.Dataset.from_generator(build_gen, (tf.float32, tf.int32), (tf.TensorShape([3,32,32]), tf.TensorShape([self.classes])))
		ds = tf.data.Dataset.from_tensor_slices((images,tf.one_hot(labels,depth=self.classes)))
		ds = ds.map(build_gen,num_parallel_calls=4)
		ds = ds.batch(batch_size=self.batch_size)
		return ds.prefetch(self.prefetch)

	def __call__(self,mode):
		return self.__iter__(mode)

	def __iter__(self,mode):
		if mode =='train':
			for X in iter(self.train_ds.shuffle(buffer_size=self.buffer_size)):
				yield X[0],X[1]
			
		if mode =='test':
			for X in iter(self.test_ds.shuffle(buffer_size=self.buffer_size)):
				yield X[0],X[1]


class cifar_100_dataset(object):
	"""docstring for cifar_100_dataset"""
	def __init__(self, filename,classes=100,batch_size=32,buffer_size=1024,norm_type='Constant',prefetch=3):
		super(cifar_100_dataset, self).__init__()
		self.filename = filename
		self.batch_size = batch_size
		self.buffer_size = buffer_size
		self.prefetch = prefetch
		self.fetch_data()
		self.classes = classes
		self.train_ds = self.datasetGenerator(self.train_img_data,self.train_lbl_data)
		self.test_ds = self.datasetGenerator(self.test_img_data,self.test_lbl_data)
		self.mode = 'train'

	def fetch_data(self):
		self.train_img_data = []
		self.train_lbl_data = []
		self.test_img_data = []
		self.test_lbl_data = []
		with open(self.filename+'train', 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
			train_img_data = np.transpose(dict[b'data'].reshape(-1,3,32,32),(0,2,3,1))
			self.train_lbl_data = np.asarray(dict[b'fine_labels'])
		
		self.train_img_data = train_img_data

		with open(self.filename+'test', 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
			test_img_data = np.transpose(dict[b'data'].reshape(-1,3,32,32),(0,2,3,1))
			self.test_lbl_data = np.asarray(dict[b'fine_labels'])

		self.test_img_data = test_img_data
		del train_img_data,test_img_data	

	
	
	def datasetGenerator(self,images,labels):
		def build_gen(img,lbl):
			return tf.image.resize_images(img,size=(227,227))/255.0,lbl
		# ds = tf.data.Dataset.from_generator(build_gen, (tf.float32, tf.int32), (tf.TensorShape([3,32,32]), tf.TensorShape([self.classes])))
		ds = tf.data.Dataset.from_tensor_slices((images,tf.one_hot(labels,depth=self.classes)))
		ds = ds.map(build_gen,num_parallel_calls=4)
		ds = ds.batch(batch_size=self.batch_size)
		return ds.prefetch(self.prefetch)

	def __call__(self,mode):
		return self.__iter__(mode)

	def __iter__(self,mode):
		if mode =='train':
			for X in iter(self.train_ds.shuffle(buffer_size=self.buffer_size)):
				yield X[0],X[1]
			
		if mode =='test':
			for X in iter(self.test_ds.shuffle(buffer_size=self.buffer_size)):
				yield X[0],X[1]
		

class shvn_dataset(object):
	"""docstring for cifar_100_dataset"""
	def __init__(self, filename,classes=10,batch_size=32,buffer_size=1024,norm_type='Constant',prefetch=1):
		super(shvn_dataset, self).__init__()
		self.filename = filename
		self.batch_size = batch_size
		self.buffer_size = buffer_size
		self.prefetch = prefetch
		self.fetch_data()
		self.classes = classes
		self.train_ds = self.datasetGenerator(self.train_img_data,self.train_lbl_data)
		self.test_ds = self.datasetGenerator(self.test_img_data,self.test_lbl_data)
		self.mode = 'train'

	def fetch_data(self):
		train_img_data = []
		train_lbl_data = []
		test_img_data = []
		test_lbl_data = []

		mat = scipy.io.loadmat(self.filename+'train_32x32.mat')
		self.train_img_data = np.transpose(mat['X'],(3,0,1,2))/255.0
		self.train_lbl_data = np.ravel(mat['y'])

		# self.train_img_data,self.train_lbl_data = shuffle(self.train_img_data,self.train_lbl_data)
		
		mat = scipy.io.loadmat(self.filename+'test_32x32.mat')
		self.test_img_data = np.transpose(mat['X'],(3,0,1,2))/255.0
		self.test_lbl_data = np.ravel(mat['y'])

		# self.test_img_data,self.test_lbl_data = shuffle(self.test_img_data,self.test_lbl_data)
		del mat

	
	def datasetGenerator(self,images,labels):
		def build_gen(img,lbl):
			return tf.image.resize_images(img,size=(227,227)),lbl
		# ds = tf.data.Dataset.from_generator(build_gen, (tf.float32, tf.int32), (tf.TensorShape([3,32,32]), tf.TensorShape([self.classes])))
		ds = tf.data.Dataset.from_tensor_slices((images,tf.one_hot(labels,depth=self.classes)))
		ds = ds.map(build_gen,num_parallel_calls=4)
		ds = ds.batch(batch_size=self.batch_size)
		return ds.prefetch(self.prefetch)

	def __call__(self,mode):
		return self.__iter__(mode)

	def __iter__(self,mode):
		if mode =='train':
			for X in iter(self.train_ds.shuffle(buffer_size=self.buffer_size)):
				yield X[0],X[1]
			
		if mode =='test':
			for X in iter(self.test_ds.shuffle(buffer_size=self.buffer_size)):
				yield X[0],X[1]


class cub_dataset(object):
	"""docstring for cifar_100_dataset"""
	def __init__(self, filename,classes=200,batch_size=32,buffer_size=1024,norm_type='Constant',prefetch=1):
		super(cub_dataset, self).__init__()
		self.filename = filename
		self.batch_size = batch_size
		self.buffer_size = buffer_size
		self.prefetch = prefetch
		self.fetch_data()
		self.classes = classes
		self.train_ds = self.datasetGenerator(self.train_img_data,self.train_lbl_data)
		self.test_ds = self.datasetGenerator(self.test_img_data,self.test_lbl_data)
		self.mode = 'train'

	def fetch_data(self):
		train_img_data = []
		train_lbl_data = []
		test_img_data = []
		test_lbl_data = []

		self.train_img_data = np.load(self.filename+'train_X.npy')
		self.train_lbl_data = np.load(self.filename+'train_Y.npy').reshape(-1,1)
		self.train_img_data,self.train_lbl_data = shuffle(self.train_img_data,self.train_lbl_data)
		
		self.test_img_data = np.load(self.filename+'test_X.npy')
		self.test_lbl_data = np.load(self.filename+'test_Y.npy').reshape(-1,1)
		self.test_img_data,self.test_lbl_data = shuffle(self.test_img_data,self.test_lbl_data)

	
	def datasetGenerator(self,images,labels):
		# ds = tf.data.Dataset.from_generator(build_gen, (tf.float32, tf.int32), (tf.TensorShape([3,32,32]), tf.TensorShape([self.classes])))
		ds = tf.data.Dataset.from_tensor_slices((images,tf.one_hot(labels,depth=self.classes)))
		ds = ds.batch(batch_size=self.batch_size)
		return ds.prefetch(self.prefetch)

	def __call__(self,mode):
		return self.__iter__(mode)

	def __iter__(self,mode):
		if mode =='train':
			for X in iter(self.train_ds.shuffle(buffer_size=self.buffer_size)):
				yield X[0],X[1]
			
		if mode =='test':
			for X in iter(self.test_ds.shuffle(buffer_size=self.buffer_size)):
				yield X[0],X[1]
		