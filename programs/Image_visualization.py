#!/usr/bin/env python
import importlib
import model
import loss
import data_loader
import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.enable_eager_execution()
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.misc import imsave
from scipy.cluster.vq import whiten

lr=0.0000
opt = tf.train.AdamOptimizer(learning_rate=lr)

dataset_id = sys.argv[1]
model_dir = sys.argv[2]
data_dir = sys.argv[3]
result_dir = sys.argv[4]
criterian = loss.softmax_cross_entropy()


if dataset_id =='cifar10':
    model_spatial_name = 'AlexNetAttPCSpatial.h5'
    model_spatial_2_name = 'AlexNetAttDSSpatial.h5'
    model_simple_name = 'AlexNetAttGAP.h5'
    classes = 10
    dataset = data_loader.cifar_10_dataset(data_dir+'/cifar-10-batches-py/', buffer_size=1, batch_size=128)

elif dataset_id =='cifar100':
    classes = 100
    model_spatial_name = 'AlexNetAttPCSpatial.h5'
    model_spatial_2_name = 'AlexNetAttDSSpatial.h5'
    model_simple_name = 'AlexNetAttGAP.h5'
    dataset = data_loader.cifar_100_dataset(
        data_dir+'/cifar-100-python/', buffer_size=1, batch_size=128)


elif dataset_id =='shvn':
    classes = 10
    model_spatial_name = 'AlexNetAttPCSpatial.h5'
    model_spatial_2_name = 'AlexNetAttDSSpatial.h5'
    model_simple_name = 'AlexNetAttGAP.h5'
    dataset = data_loader.shvn_dataset(
        data_dir+'/SHVN/', buffer_size=1, batch_size=128, classes=10)

output_gap_path = '%s/%s/gap/' % (result_dir,dataset_id)
os.makedirs(output_gap_path,exist_ok=True)
output_spatial_path = '%s/%s/spatialPC/' % (result_dir,dataset_id)
os.makedirs(output_spatial_path,exist_ok=True)
output_spatial_2_path = '%s/%s/spatialDS/'%(result_dir, dataset_id)
os.makedirs(output_spatial_2_path,exist_ok=True)
model_root = model_dir+'/'

cmap = plt.get_cmap('jet')
(x,y)= next(dataset('train'))
print(x.shape,y.shape)
def load_spatial_model(x):
    model_path = model_root+model_spatial_name
    alexattnet = model.AlexNetAttSpatial(classes=classes)
    alexattnet.compile(optimizer=opt,loss=criterian)
    p_acc = 0
    epoch_loss = 0
    alexattnet(x)
    alexattnet.load_weights(model_path)
    alexattnet.p=0
    return alexattnet


def load_spatial_2_model(x):
    model_path = model_root+model_spatial_2_name
    alexattnet = model.AlexAttNet(classes=classes)
    alexattnet.compile(optimizer=opt,loss=criterian)
    p_acc = 0
    epoch_loss = 0
    alexattnet(x)
    alexattnet.load_weights(model_path)
    alexattnet.p=0
    return alexattnet

def load_simple_model(x):
    model_path = model_root+model_simple_name
    alexnet = model.AlexNetGAP(classes=classes)
    alexnet.compile(optimizer=opt,loss=criterian)
    p_acc = 0
    epoch_loss = 0
    alexnet(x)
    alexnet.load_weights(model_path)
    alexnet.p=0
    return alexnet


# def whiten(x):
#     x = x
#     # std = np.std(x)
#     return x

def print_att_maps(x,y,k,output_spatial_path):
    (conv4,conv5)= alexattnet(x)
    conv4 = conv4.numpy()
    conv5 = conv5.numpy()
    for j,x_img in enumerate(x):

        x_conv4 = whiten(resize(conv4[j].reshape((13,13)),(227,227)).reshape(-1,1)).reshape(227,227)
        x_conv5 = whiten(resize(conv5[j].reshape((13,13)),(227,227)).reshape(-1,1)).reshape(227,227)
        conv = x_conv4*0.5+x_conv5*0.5
        p = np.percentile(conv,60)
        mask = np.zeros((conv.shape[0],conv.shape[1],3),dtype=np.uint8)
        mask[conv>=p,:] = 1.0
        
        imgs = [x_img]
        for i,heat_img in enumerate([x_conv4,x_conv5]):
            rgb_img = np.delete(cmap(heat_img), 3, 2)
            imgs.append(rgb_img)

        imgs.append(0.4*x_img+0.6*np.delete(cmap(0.5*x_conv4+0.5*x_conv5), 3, 2))
        imgs[1] = 0.5*x_img+0.5*imgs[1]
        imgs[2] = 0.5*x_img+0.5*imgs[2]
        imgs.append(0.5*x_img+0.5*mask)
        imgs = np.hstack(imgs)
        imsave(output_spatial_path+'%d.png'%(k+1),imgs)
        k+=1


def print_gap_maps(x,y,k):
    convs= alexnet(x).numpy()
    for j,x_img in enumerate(x):
        conv = whiten(convs[j].reshape(-1,1)).reshape((13,13))
        conv = conv/np.max(conv)
        conv = resize(conv,(227,227))
        p = np.percentile(conv,50)
        mask = np.zeros((conv.shape[0],conv.shape[1],3),dtype=np.uint8)
        mask[conv>=p,:] = 1.0
        imgs = [x_img]
        rgb_img = np.delete(cmap(conv), 3, 2)
        imgs.append(rgb_img)
        imgs[1] = 0.5*x_img+0.5*imgs[1]
        imgs.append(0.5*x_img+0.5*mask)
        imgs = np.hstack(imgs)

        imsave(output_gap_path+'%d.png'%(k+1),imgs)
        k+=1

alexattnet = load_spatial_model(x)
print_att_maps(x,y,0,output_spatial_path)
del alexattnet

alexattnet = load_spatial_2_model(x)
print_att_maps(x,y,0,output_spatial_2_path)
del alexattnet

alexnet = load_simple_model(x)
print_gap_maps(x,y,0)
del alexnet
