#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:05:23 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;

import scipy.io as scio;
import matplotlib.image as img;
import matplotlib.pyplot as plt;
import matplotlib.gridspec as gs;
from autoencoder_trees import *;

a = scio.loadmat("mnist_all.mat");
d = (28,28);

D = 10;
I = 75;
E_train = np.zeros((D*I,28*28));
T_train = np.zeros((D*I,10));

E_test = np.zeros((2*D*I,28*28));
T_test = np.zeros((2*D*I,10));
for b in range(D):
    for i in range(I):
        E_train[I*b+i] =  a['train'+str(b)][i,:]/255.0;
        T_train[I*b+i][b] = 1.0;

for i in range(2*I):
    for b in range(D):
        E_test[D*i+b]    = a['test'+str(b)][i,:]/255.0;
        T_test[D*i+b][b] = 1.0;

L = 6;
tree = ae_tree(E_train,T_train,3,L);

it = 0;
tree.it = 0;
while it < 1000:
    print it;
    tree.step(100,1e-1);
    it += 1;

f = plt.figure();
gs_ = gs.GridSpec(L,2**(L-1));
a = 0;
for l in range(L-1):
    for k in range(2**l):
        aux_ax = f.add_subplot(gs_[l,k]);
        aux_ax.imshow(tree.w[a+k].reshape(28,28),cmap='jet');
        aux_ax.set_xticklabels([]);
        aux_ax.set_yticklabels([]);
        aux_ax.grid(False)
    a += 2**l;
for k in range(2**(L-1)):
    aux = np.zeros((28,28,3));
    aux[:,:,0] = tree.v[a+k][0,:].reshape(28,28);
    aux[:,:,1] = tree.v[a+k][1,:].reshape(28,28);
    aux[:,:,2] = tree.v[a+k][2,:].reshape(28,28);
    aux -= np.min(aux);
    aux /= np.max(aux)+1e-10;
    aux_ax = f.add_subplot(gs_[L-1,k]);
    aux_ax.imshow(aux);
    aux_ax.set_xticklabels([]);
    aux_ax.set_yticklabels([]);
    aux_ax.grid(False)
plt.show();

Y = np.zeros((E_test.shape[0],3));
C = np.zeros(E_test.shape[0],dtype=int)
for i in range(E_test.shape[0]):
    Y[i] = tree.eval_y(E_test[i],0);
    C[i] = np.argmax(T_test[i]);

from mpl_toolkits.mplot3d import Axes3D;

fig = plt.figure();
ax = fig.add_subplot(111, projection='3d');
ax.scatter(Y[:,0],Y[:,1],Y[:,2],c=C,cmap='jet')