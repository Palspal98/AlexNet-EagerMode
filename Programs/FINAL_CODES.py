#!/usr/bin/env python
# coding: utf-8

# In[1]:


import importlib
import model
import loss
import h5py
import data_loader
import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.enable_eager_execution()


# In[2]:


import sys
import time

root = './data_t'
model_path = root+'/model/'

if sys.argv[2]=='cifar10':
    lr=1e-5
    dataset = data_loader.cifar_10_dataset(root+'/dataset/cifar-10-batches-py/',buffer_size=1024,batch_size=128)
    if sys.argv[1]=='0':
        f = open(root+'/logs/AlexNet_cifar10.log','w')
        model_name = 'AlexNetAtt_cifar10'
        alexattnet = model.AlexNet(classes=10)

    if sys.argv[1]=='1':
        f = open(root+'/logs/AlexNetAttDSSpatial_cifar10.log','w')
        model_name = 'AlexNetAttDSSpatial_cifar10'
        alexattnet = model.AlexAttNet(classes=10)

    elif sys.argv[1]=='2':
        f = open(root+'/logs/AlexNetAttPCSpatial_cifar10.log','w')
        model_name = 'AlexNetAttPCSpatial_cifar10'
        alexattnet = model.AlexNetAttSpatial(classes=10)
    
    elif sys.argv[1]=='3':
        f = open(root+'/logs/AlexNetGAP_cifar10.log','w')
        model_name = 'AlexNetAttGAP_cifar10'
        alexattnet = model.AlexNetGAP(classes=10)

elif sys.argv[2]=='cifar100':
    lr=1e-5
    dataset = data_loader.cifar_100_dataset(root+'/dataset/cifar-100-python/',buffer_size=1024,batch_size=128)
    if sys.argv[1]=='0':
        f = open(root+'/logs/AlexNet_cifar100.log','w')
        model_name = 'AlexNetAtt_cifar100'
        alexattnet = model.AlexNet(classes=100)

    if sys.argv[1]=='1':
        f = open(root+'/logs/AlexNetAttDSSpatial_cifar100.log','w')
        model_name = 'AlexNetAttDSSpatial_cifar100'
        alexattnet = model.AlexAttNet(classes=100)

    elif sys.argv[1]=='2':
        f = open(root+'/logs/AlexNetAttPCSpatial_cifar100.log','w')
        model_name = 'AlexNetAttPCSpatial_cifar100'
        alexattnet = model.AlexNetAttSpatial(classes=100)
    
    elif sys.argv[1]=='3':
        f = open(root+'/logs/AlexNetGAP_cifar100.log','w')
        model_name = 'AlexNetAttGAP_cifar100'
        alexattnet = model.AlexNetGAP(classes=100)

elif sys.argv[2]=='shvn':
    lr=1e-6
    dataset = data_loader.shvn_dataset(root+'/dataset/SHVN/',buffer_size=1,batch_size=128,classes=10,prefetch=3)
    if sys.argv[1]=='0':
        f = open(root+'/logs/AlexNet_shvn.log','w')
        model_name = 'AlexNetAtt_shvn'
        alexattnet = model.AlexNet(classes=10)

    if sys.argv[1]=='1':
        f = open(root+'/logs/AlexNetAttDSSpatial_shvn.log','w')
        model_name = 'AlexNetAttDSSpatial_shvn'
        alexattnet = model.AlexAttNet(classes=10)

    elif sys.argv[1]=='2':
        f = open(root+'/logs/AlexNetAttPCSpatial_shvn.log','w')
        model_name = 'AlexNetAttPCSpatial_shvn'
        alexattnet = model.AlexNetAttSpatial(classes=10)

    elif sys.argv[1]=='3':
        f = open(root+'/logs/AlexNetGAP_shvn.log','w')
        model_name = 'AlexNetAttGAP_shvn'
        alexattnet = model.AlexNetGAP(classes=10)


elif sys.argv[2]=='cub':
    lr=1e-5
    dataset = data_loader.cub_dataset(root+'/dataset/CUB_200_2011/',buffer_size=256,batch_size=128,classes=200,prefetch=3)
    if sys.argv[1]=='0':
        f = open(root+'/logs/AlexNet_cub.log','w')
        model_name = 'AlexNetAtt_cub'
        alexattnet = model.AlexNet(classes=200)

    if sys.argv[1]=='1':
        f = open(root+'/logs/AlexNetAttDSSpatial_cub.log','w')
        model_name = 'AlexNetAttDSSpatial_cub'
        alexattnet = model.AlexAttNet(classes=200)

    elif sys.argv[1]=='2':
        f = open(root+'/logs/AlexNetAttPCSpatial_cub.log','w')
        model_name = 'AlexNetAttPCSpatial_cub'
        alexattnet = model.AlexNetAttSpatial(classes=200)

def log_print(msg):
    sys.__stdout__.write("Time elapsed %s: %s\n"%(time.ctime(),msg))
    print('Time elapsed %s: %s'%(time.ctime(),msg),flush=True,file=f)

def calc_Accuracy(model,dataset):
    accuracy = 0.0
    total_points = 0.0
    for i,(x,y) in enumerate(dataset('test')):
        y = np.argmax(y.numpy(),axis=1).reshape(-1,1)
        yhat = np.argmax(model.predict(x),axis=1).reshape(-1,1)
        total_points+=y.shape[0]
        col_idx = np.arange(y.shape[0])
        accuracy+=np.sum(y==yhat)
        print('Evaluating %d'%(i),end='\r')
    return accuracy/total_points


Epochs = 300
opt = tf.train.AdamOptimizer(learning_rate=lr)
criterian = loss.softmax_cross_entropy()
alexattnet.compile(optimizer=opt,loss=criterian)
p_acc = 0
for e in range(Epochs):
    epoch_loss = 0
    for i,(x,y) in enumerate(dataset('train')):
        epoch_loss+=alexattnet.fit(x=x,y=y,epochs=1,verbose=0,batch_size=128).history['loss'][0]
        print('Epoch %d batch %d'%(e+1,i+1),end='\r')
    if (e+1)%25 ==0:
        opt.__dict__['_lr']/=2
    if (e+1)%2 == 0:
        accuracy = calc_Accuracy(alexattnet,dataset)
        if accuracy > p_acc:
            alexattnet.save_weights(filepath=model_path+model_name+'.h5')
            p_acc = accuracy
        log_print("Epoch %d, loss %f acc %f"%(e+1,epoch_loss,accuracy))
        continue
    log_print('Epoch %d loss %f'%(e+1,epoch_loss))


# In[7]:
alexattnet.summary()
