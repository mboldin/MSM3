##  MSM estimation
##    Test case Unemployment Quarterly model
##    M Boldin   June/July 2018

## Expected results
##        MSM estimation
##        2018-07-05 16:04:01.462451
##        Starting Values: [0.0073141794305075225, 0.1074581804529243, 0.9865332520406973, 0.9865332520406973, 0.2707890811252287, 0.2707890811252287, 0.7, 0.3]  
##           LogLik 0: -13.61713055418656
##        Results-- Llv: -39.62150572085464 | Method: BFGS          Status:    2 | 
##        Iter:   28 Evals: 1222 -- Time: 1.8 seconds
##        Desired error not necessarily achieved due to precision loss.
##        Estimated values: [0.03997804 0.11060476 0.97291618 1.03896481 0.13088873 0.31906573
##         0.96988698 0.12478534]
##
##        Bs
##        [[0.03997804 0.11060476]
##         [0.97291618 1.03896481]]
##        sees
##        [[0.13088873 0.31906573]]
##        Q
##        [[0.96988698 0.03011302]
##         [0.12478534 0.87521466]]
##        po
##        [[0.80559496 0.19440504]]
##        39.628956173929645
##        >>>

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
def OLScalc(y,X):
    """Compute OLS parameters, includ LogLike
    """
    
    y = matrix(y)
    if y.shape[0] == 1:
        y = y.T
    X = matrix(X)
    k = X.shape[1]
    nobs = len(y)
    
    iXtX= (X.T*X).I
    B = (iXtX)*(X.T*y)
    
    e = y - X*B    
    see = np.std(e)
    
    C = (iXtX)*see**2
    Bse = np.sqrt(C.diagonal()).T

    R2  = 1 - e.var()/y.var()

    ##Likelihood calc
    a = 1 / ( sqrt(2*np.pi)*see )
    e2 = np.power(e,2)/see**2
    f = a*np.exp(-.5*e2)
    lpt = 1
    for t in range(nobs):
        lpt = lpt * f[t]
    llv = log(lpt)

    print('OLS calc')
    M = np.zeros((k,2))
    M[:,0]= B.T
    M[:,1]= Bse.T
    print(M)
    print(nobs, see, R2, llv)
    return (B, Bse, see)


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
        results = minimize(m.msmloglike, svalues, method=method, tol=1E-8)
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


def run_minimize(svalues= None, f= None, method= 'BFGS'):

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

###################################################

## Data Setup for Unemployment model

ddir1= r'/home/michael/Desktop/PyProg/MSM/Set3'
dx = pd.read_excel( os.path.join(ddir1,'unrate.xlsx') )
#print(dx)

Yvar = 'UNRATE'
Xvar = ['C','lag']
y = dx.loc[:,'UNRATE']
nobs =len(y)

datx = pd.DataFrame(y)
datx['C'] = ones((nobs,1))
datx['lag'] = dx['lag']

y = matrix(datx.loc[:,Yvar]).transpose()
X = matrix(datx.loc[:,Xvar])


## MSM setup check
ns = 2
k = 2
bh1 = [-0.1, 0.1, 1.0, 1.0, .30 , .30, .80, 0.2 ]
bhx = array(bh1)

method = 'BFGS'

#####################################################################


if 1:

    ## OLS case
    y, X = (datx[Yvar], datx[Xvar])
    (B, Bse, see) = OLScalc(y,X)

if 1:

    print()
    m = msmset1(ns=2, k=2, y=y, X=X)    

    rmse = see.item()
    adj = 0.5*Bse[0].item()
    sv1a = [ B[0].item() - adj, B[0].item() + adj, B[1].item(), B[1].item(), rmse, rmse, .70, .30 ]
    sv1b = [0.3, 0.8, 0.90, 0.90, .25 , .25, .90, .10 ]

    run_minimize(svalues= sv1a, method=method)
    #ShowExtra(m)

if 0:
    datx2 = pd.DataFrame(dx)
    datx2.set_index('Date', inplace=True)
    unrate = datx2.loc[:,'UNRATE']
    lag = datx2.loc[:,'lag']
    msm2 = sm.tsa.MarkovRegression(endog=unrate, exog=lag, k_regimes=2, switching_exog=True, switching_variance=True)

    #results1 = msm2.fit(method = 'bfgs')
    #print( results1.summary())

    sv2 = [ .50, .50, B[0] - adj, B[0] + adj, B[1], B[1], rmse**2, rmse**2 ]
    results2 = msm2.fit(start_params= sv2, method = 'bfgs')
    print( results2.summary())
    results2 = msm2.fit(start_params= sv2, order=2, method = 'bfgs')
    print( results2.summary())

    #sv3 = [  0.974,    0.213,  -0.033,  -0.836,    0.991,    1.251,    0.03,  0.03 ]   
    #results3 = msm2.fit(start_params= sv3, method = 'bfgs')
    #print( results3.summary())


if 0:

    print()
    y = matrix(datx.loc[:,Yvar]).transpose()
    X = matrix(datx.loc[:,Xvar])
    
    B = results0.params
    Bse = results0.bse
    rmse = results0.mse_resid**.5
    adj = 0.1*Bse[0]
    rmse = results0.mse_resid**.5
    bhx = [ B[0] - adj, B[0] + adj, B[1], B[1], rmse, rmse, .50, .50 ]
    bhx = [0.3, 0.8, 0.90, 0.90, .25 , .25, .90, .10 ]
    #bhx = [0.13704220855276644, 0.0969433157605557, 1.171131531783023, 1.14096519373727, 0.282600849714314, 0.18910405164493635, 0.9452539913014765, 0.16063851485499858] 
    bhx = [0.039719795598223445, 0.11619930235234315, 0.9116977999149455, 0.9563886231411491, 0.12459361383176258, 0.3481463643972497, 0.9688856291695168, 0.11814517402945801]

    m = msmset1(ns=2, k=2, y=y, X=X)    
    m.update(bhx)
    m.calcllv2()
    px= (bhx, m.llv)
    msg = 'Starting Values: %s  \n LL value: %s' % px
    print( msg )
    print( msmlik(bhx), m.llv )

    dt1= dt.now() 
    results = minimize(msmlik, bhx, method= method, tol=1E-2)
    timex = dt.now() - dt1
    timex = timex.seconds + timex.microseconds/1E6

    px= (method, results.fun, results.nit, timex, results.status,)
    msg = 'MSM Estimation | Method: %-10s  -LLV: %8s | Iter: %4d  Time: %.1f seconds | Status: %4s ' % px
    print( msg )
    if not results.status == 0:
        print(results.message)
    print(results.x)
    

if 0:
    ## Time LLV calc
    Bs = m.Bs
    sees = m.sees
    Q = m.Q
    po = lpo(Q)
    e = y - X*Bs    
    nobs = len(e)
    a = 1 / ( sqrt(2*np.pi)*sees )
    a = repmat(a, nobs) 
    e2 = np.power(e/sees,2)
    f1 = np.exp(-.5*e2)
    Fs =  np.multiply(a,f1) 
    dt1= dt.now()
    N = 1
    for k in range(N):
        llv1 = MSMllv1(Fs, Q, po)
        lpt = MSMllv2b(Fs, Q, po)
        llv2 = log(np.sum(lpt))
        #print(k,m.llv)
    timex = dt.now() - dt1
    timex = timex.seconds + timex.microseconds/1E6
    print(timex, k, m.llv, llv1, llv2)


