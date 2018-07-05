##  MSM estimation
##    Test case Version 2 Unemployment Quarterly model
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

from pprint import pprint

import numpy as np
from numpy import array, matrix
import pandas as pd

import scipy
from scipy import stats
from scipy.optimize import minimize

from MSM3a import  *

##################################################

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

##################################################################

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
method = 'BFGS'

#####################################################################


if 1:
    ## OLS case
    y, X = (datx[Yvar], datx[Xvar])
    (B, Bse, see) = OLScalc(y,X)

if 1:
    print()
    m = msmset1(ns=2, k=2, y=y, X=X)    
    #rmse = see.item()
    #adj = 0.5*Bse[0].item()
    #sv1a = [ B[0].item() - adj, B[0].item() + adj, B[1].item(), B[1].item(), rmse, rmse, .70, .30 ]
    sv1b = [0.3, 0.8, 0.90, 0.90, .25 , .25, .90, .10 ]

    run_minimize(m, svalues= sv1b, method=method)
    #ShowExtra(m)

