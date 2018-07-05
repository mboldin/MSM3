##  MSM set up and estimation
##    M Boldin   June/July 2018

#################################################################

import os
import datetime as dt

import random as rm
import pickle

from pprint import pprint

from math import sqrt, cos, sin, exp, log, pi
import numpy as np
from numpy import array, matrix
from numpy import zeros, ones, eye, diag, kron

import pandas as pd

import scipy
from scipy import stats
from scipy.optimize import minimize

from numba import jit, autojit 

    
##################################################################

## MATLAB like functions

##def ones(nr,nc):
##    msize= makeRowColtuple(nr,nc)
##    return np.ones(msize)
##
##def zeros(nr,nc):
##    msize= makeRowColtuple(nr,nc)
##    return np.zeros(msize)
#def t(a):
#    return matrix(a).transpose()

def vech(a):
    return a.flatten(1).transpose()

def reshape(a,r=None,c=None):
    a= matrix(a)
    nn= a.size
    if c==1:
        a=a.reshape((nn,1))
    elif r==1:
        a=a.reshape((1,nn))
    elif r>0 and c>0:
        a=a.reshape((r,c))
    elif len(r)==2:
        a=a.reshape(r)
    else:
        a=a.reshape((r,c))
    return a

def repmat(a,r=1,c=1):
    return kron(np.ones((r,c)),a)

##################################################################


def MSMloglike(Bs,sees,Q, po, y, X):

    e = y - X*Bs    
    nobs = len(e)
    
    a = 1 / ( sqrt(2*np.pi)*sees )
    a = repmat(a, nobs) 
    #print(e)
    #e2 = 1
    e2 = np.power(e/sees,2)
    #print(e2)
    f1 = np.exp(-.5*e2)
    Fs =  np.multiply(a,f1) 
    try:
        vofadj = 1
        lpt = vofadj * po 
        for t in range(nobs):
            #print (t, y[t], e[t], Fs[t])
            lpt = np.multiply((lpt * Q), Fs[t])
            #print (t, lpt, log(np.sum(lpt)) )
        llv = log(np.sum(lpt))
        #print( llv, llv / vofadj)
    except ValueError:
        llv= -1E6 
    #v = llv / vofadj 
    self.llv = llv
    
def MSMllv1(Fs, Q, po):
    nobs = len(Fs)
    lpt = matrix(po)
    for t in range(nobs):
        lpt = np.multiply((lpt * Q), Fs[t])
    llv = log(np.sum(lpt))
    return llv


@jit
def msmcalc2a(Fs, Q, po):
    nobs = len(Fs)
    lpt = 1 * po
    ns = lpt.shape[1]
    for t in range(nobs):
        l0 = 1 * lpt
        for j in range(ns):
            lpt[0,j] = 0
            for k in range(ns): 
                lpt[0,j] = lpt[0,j] + (l0[0,k] * Q[j,k])
            lpt[0,j] = lpt[0,j] * Fs[t,j]
    return lpt

@jit
def msmcalc2b(Fs, Q, po):
    nobs = len(Fs)
    lpt = 1 * po
    for t in range(nobs):
        l0, l1 = lpt[0,0], lpt[0,1] 
        lpt[0,0] = ( l0 *Q[0,0] + l1 * Q[1,0]) * Fs[t,0]
        lpt[0,1] = ( l0 * Q[0,1] + l1 * Q[1,1]) * Fs[t,1]
    return lpt


def msmset2(bh):
    """ Alternative set up of MSM parameters """
        ##def update1(self, bh, u1, u2):
    bh = array(bh)
    k, ns = self.k, self.ns
    Bs = zeros(k,ns)
    Bs[1,0] = bh[0]  
    Bs[1,1] = bh[1]  
    Bs[0,0] = u1 * (1-Bs[1,0])  
    Bs[0,1] = u2 * (1-Bs[1,1])  
    self.Bs = Bs
    sees = zeros(1,ns)
    sees[0,0] = bh[2]
    sees[0,1] = bh[3]
    self.sees = matrix(abs(sees))
    q1 = zeros(ns,1)
    q1[0,0] = bh[4]
    #q1[1,0] = 1-bh[2]
    q1[1,0] = bh[5]
    Q = rq(q1)
    self.Q = Q
    self.po = lpo(self.Q)
    return (Bs,see,Q,po)


class msmset1():
    """ MSM set up
       init: ns=2, k=1, y=None, X=None
       attributes created by update()
          Bs, sees
    """

    def __init__(self, ns=2, k=1, y=None, X=None):
        self.k = k
        self.ns = ns
        self.y = matrix(y).T
        self.X = matrix(X)
        
    def update(self,x):
        bh = array(x)
        k, ns = self.k, self.ns
        Bs = matrix( x[:ns*k] )  
        self.Bs = reshape(Bs,k,ns)
        sees = matrix( x[ns*k:ns*k+ns] )
        self.sees = abs(sees)
        q1 = matrix( x[ns*k+ns:] )
        self.Q = self.rq( q1.transpose() )
        self.po = self.lpo(self.Q)
        self.llv = None
        e = self.y - self.X*self.Bs    
        nobs = len(e)
        a = 1 / ( sqrt(2*np.pi)*sees )
        a = repmat(a, nobs) 
        e2 = np.power(e/sees,2)
        f1 = np.exp(-.5*e2)
        self.Fs =  np.multiply(a,f1) 

    def lpo(self,Q):
        Q= matrix(Q)
        ns, ns2 = Q.shape
        po= zeros((1,ns))
        if (ns == 2):
            dn= Q[1,0]+Q[0,1]
            po[0,0]= Q[1,0] / dn 
            po[0,1]= 1 - po[0,0]
        return po

    def rq(self,xq):
        # Transforms Q matrix to satisfy transition matrix requirements
        #   Part of MSM function group (MBoldin Oct 2002)
        #   Forces each element to be between 0-1 and make rows add to 1
        #        q = xq ./ ( sum(xq')'*ones(1,nc) )
        #   arg2=1 turns n x n-1 into n x n matrix
        xq = matrix(xq)
        nr,nc = xq.shape
        if nc > nr:
            xq = xq.transpose()
            nr,nc = xq.shape
        Q= zeros((nr,nr))
        Q[:,0:]= xq[0:]
        if (nc == nr-1):
            Q[:,-1:]=(1-np.sum(xq,axis=1))
            nc=nc+1
        Q=abs(Q);
        #h= t(matrix(np.sum(Q,axis=1))) * ones((1,nc))
        h= matrix(np.sum(Q,axis=1)).transpose() * ones((1,nc)) 
        Q = np.divide(Q,h); 
        return Q



    def loglike(self):

        Bs, sees, Q, po = self.Bs, self.sees, self.Q, self.po
        e = self.y - self.X*self.Bs    
        nobs = len(e)
        
        a = 1 / ( sqrt(2*np.pi)*sees )
        a = repmat(a, nobs) 
        e2 = np.power(e/sees,2)
        f1 = np.exp(-.5*e2)
        Fs =  np.multiply(a,f1) 
        try:
            vofadj = 1
            lpt = vofadj * po 
            for t in range(nobs):
                lpt = np.multiply((lpt * Q), Fs[t])
            llv = log(np.sum(lpt))
        except ValueError:
            llv= -1E6 
        self.llv = llv


    def llcalc(self):
        e = self.y - self.X*self.Bs    
        nobs = len(e)
        e2 = np.power(e/self.sees,2)
        a = 1 / ( sqrt(2*np.pi)*self.sees )
        a = repmat(a, nobs) 
        self.Fs =  np.multiply(a,np.exp(-.5*e2)) 
        vofadj = 1
        ## Using numba @jit optimized function
        lpt = msmcalc2a(self.Fs, self.Q, self.po)
        llv = log(lpt.sum())
        return llv

    def msmloglike(self, params):
        self.update(params)
        try:
            llv = self.llcalc()
        except:
            llv = -1E6
        return -llv

    def maxlik(self, svalues, method):
        #results = minimize(m.msmlik, svalues, method=method, tol=1E-8)
        results = minimize(self.msmloglike, svalues, method=method, tol=1E-8)
        self.results = results



    def show_results(self):
        print('Bs')
        print(self.Bs)
        print('sees')
        print(self.sees)
        print('Q')
        print(self.Q)
        print('po')
        print(self.po)
        print(self.llv)

    def msmcalc(self):
        Q = self.Q
        nobs = len(self.Fs)
        lpt = 1 * self.po
        for t in range(nobs):
            lpt = np.multiply((lpt * Q), self.Fs[t])
        return lpt

    def msmcalc2(self):
        Q = self.Q
        nobs = len(self.Fs)
        lpt = 1 * self.po
        for t in range(nobs):
            l0, l1 = lpt[0,0], lpt[0,1] 
            lpt[0,0] = ( l0 *Q[0,0] + l1 * Q[1,0]) * self.Fs[t,0]
            lpt[0,1] = ( l0 * Q[0,1] + l1 * Q[1,1]) * self.Fs[t,1]
        return lpt


def run_minimize(m, svalues= None, f= None, method= 'BFGS'):

    print('MSM estimation')
    print( dt.datetime.now() )

    m.update(svalues)
    m.loglike()
    px= (svalues, m.llv)
    msg1 = 'Starting Values: %s  \n   LogLik 0: %s' % px
    print( msg1 )
    #print (m.msmcalc())

    dt1= dt.datetime.now()
    m.maxlik(svalues,method)
    results = m.results
    timex = dt.datetime.now() - dt1
    timex = timex.seconds + timex.microseconds/1E6
    px= (results.fun, method, results.status, results.nit, results.nfev, timex)
    msg2 = 'Results-- Llv: %8s | Method: %-12s  Status: %4s | \nIter: %4d Evals: %4d -- Time: %.1f seconds' % px
    print( msg2 )
    if not (results.status == 0):
        print(results.message)
    print('Estimated values:', results.x)
    print()    

    b = results.x
    m2 = m
    m2.update(b)
    m2.loglike()
    m2.show_results()


def ShowExtra(m):
    print()
    Bs = m.Bs.copy()
    un1 = Bs[0,0]/(1-Bs[1,0]) 
    un2 = Bs[0,1]/(1-Bs[1,1]) 
    print( 'un1: %.2f  un2: %.2f' % (un1, un2) )
    print()    
    return m

