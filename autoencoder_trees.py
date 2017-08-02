#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:27:29 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;
import matplotlib.pyplot as plt;

def make_tree(L):
    r = range((2**L)-1);
    a = [];
    for i in r:
        a.append((2*i+1>=len(r)));
    return a;
def sigm(t):
    return 0.5*(np.tanh(0.5*t)+1.0);

class ae_tree:
    def __init__(self,X,Y,d,L):
        self.X = X;
        self.Y = Y;
        
        self.d = d;
        self.N = X.shape[1];
        self.K = Y.shape[1];
        N = self.N;
        K = self.K;
        
        self.L = L;
        self.leafs = make_tree(L);
        
        self.R = range((2**L)-1);
        
        self.w  = [1e-8*rd.randn(N) if not self.leafs[l] else None for l in self.R];
        self.dw = [np.zeros(N) if not self.leafs[l] else None for l in self.R];
        self.mw = [np.zeros(N) if not self.leafs[l] else None for l in self.R];
        self.nw = [np.zeros(N) if not self.leafs[l] else None for l in self.R];
 
        self.v  = [1e-8*rd.randn(d,N)   if self.leafs[l] else None for l in self.R];
        self.dv = [np.zeros((d,N)) if self.leafs[l] else None for l in self.R];
        self.mv = [np.zeros((d,N)) if self.leafs[l] else None for l in self.R];
        self.nv = [np.zeros((d,N)) if self.leafs[l] else None for l in self.R];

        self.U  = 1e-8*rd.randn(K,d);
        self.dU = np.zeros((K,d));
        self.mU = np.zeros((K,d));
        self.nU = np.zeros((K,d));
        
        self.a  = 1e-8*rd.randn(K);
        self.da = np.zeros(K);
        self.ma = np.zeros(K);
        self.na = np.zeros(K);
        
        self.s = np.zeros((2**L)-1);
        
        self.y    = [np.zeros(d) for l in self.R];
        self.dJdy = [np.zeros(d) for l in self.R];
        
        self.mu = 0.5;
        self.nu = 0.999;
        self.it = 0.0;
        
    def eval_y(self,e,l):        
        #print l;
        if self.leafs[l]:
            self.y[l] = np.tanh(np.dot(self.v[l],e));
            return self.y[l];
        else:
            self.s[l] = sigm(np.dot(self.w[l],e));
            self.y[l] = self.s[l]*self.eval_y(e,2*l+1)+(1.0-self.s[l])*self.eval_y(e,2*l+2);
            return self.y[l];
        
    def eval_(self,e,t):
        y =  np.tanh(self.eval_y(e,0));
        z =  np.dot(self.U,y)+self.a;
        z -= np.max(z);
        z =  np.exp(z);
        z /= np.sum(z);
        return -np.dot(t,np.log(z+1e-30))/t.size;
    
    def grad_(self,e,t):
        y =  np.tanh(self.eval_y(e,0));
        z =  np.dot(self.U,y)+self.a;
        z -= np.max(z);
        z =  np.exp(z);
        z /= np.sum(z);
        
        dJdz = (z-t)/t.size;
        self.dU += np.outer(dJdz,y);
        self.da += dJdz;
        
        self.dJdy[0] = np.dot(dJdz,self.U)*(1.0-np.power(self.y[0],2));
        for l in self.R[1:]:
            k = (l-1)/2;
            if 2*k+1 == l:
                self.dJdy[l] = self.s[k]*self.dJdy[k];
            else:
                self.dJdy[l] = (1.0-self.s[k])*self.dJdy[k];
            if self.leafs[l]:
                self.dJdy[l] *= (1.0-np.power(self.y[l],2));

        for l in self.R:
            if self.leafs[l]:
                self.dv[l] += np.outer(self.dJdy[l],e);
            else:
                aux = self.s[l]*(1.0-self.s[l])*np.dot(self.dJdy[l],self.y[2*l+1]-self.y[2*l+2]);
                self.dw[l] += aux*e;
        return -np.dot(t,np.log(z+1e-30))/t.size;
    
    def step(self,B,dt):
        idx = np.arange(self.X.shape[0]);
        B_idx = rd.permutation(idx)[:B];
        self.dU.fill(0.0);
        self.da.fill(0.0);
        for l in self.R:
            self.dJdy[l].fill(0.0);
            if self.leafs[l]:
                self.dv[l].fill(0.0);
            else:
                self.dw[l].fill(0.0);
                
        err = 0.0;
        for b in B_idx:
            err += self.grad_(self.X[b],self.Y[b])/B;
        
        self.dU /= B;
        self.da /= B;
        for l in self.R:
            if self.leafs[l]:
                self.dv[l] /= B;
            else:
                self.dw[l] /= B;
        
        aux_mu = 1./(1.0-self.mu**(1.0+self.it));
        aux_nu = 1./(1.0-self.nu**(1.0+self.it));
        
        self.mU = self.mu*self.mU+(1.0-self.mu)*self.dU;
        self.nU = self.nu*self.nU+(1.0-self.nu)*np.power(self.dU,2);
        self.ma = self.mu*self.ma+(1.0-self.mu)*self.da;
        self.na = self.nu*self.na+(1.0-self.nu)*np.power(self.da,2);

        mU_ = aux_mu*self.mU;        
        nU_ = aux_nu*self.nU;
        self.U -= dt*mU_/(1e-2+np.sqrt(nU_));
        
        ma_ = aux_mu*self.ma;
        na_ = aux_nu*self.na;
        self.a -= dt*ma_/(1e-2+np.sqrt(na_));
        
        for l in self.R:
            if self.leafs[l]:
                self.mv[l] = self.mu*self.mv[l]+(1.0-self.mu)*self.dv[l];
                self.nv[l] = self.nu*self.nv[l]+(1.0-self.mu)*np.power(self.dv[l],2);
                
                mv_ = aux_mu*self.mv[l];
                nv_ = aux_nu*self.nv[l];
                self.v[l] -= dt*mv_/(1e-2+np.sqrt(nv_));
            else:
                self.mw[l] = self.mu*self.mw[l]+(1.0-self.mu)*self.dw[l];
                self.nw[l] = self.nu*self.nw[l]+(1.0-self.mu)*np.power(self.dw[l],2);
                
                mw_ = aux_mu*self.mw[l];
                nw_ = aux_nu*self.nw[l];
                self.w[l] -= dt*mw_/(1e-2+np.sqrt(nw_));
        print self.it,err;
        self.it += 1;
    