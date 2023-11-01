import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

def r2(y,y_hat):
  if len(y.shape)==1:
    y=y.reshape((-1,1))
  if len(y_hat.shape)==1:
    y_hat=y_hat.reshape((-1,1))
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)

def krr(xs,x_tr,y_tr_in,lbda,sigma, nu=np.inf,center=True):
  y_tr_mean=np.mean(y_tr_in) if center else 0
  y_tr=y_tr_in-y_tr_mean
  Ks=kern(xs,x_tr,sigma, nu)
  K=kern(x_tr,x_tr,sigma, nu)
  return Ks@np.linalg.solve(K+lbda*np.eye(K.shape[0]),y_tr)+y_tr_mean


def kern(X,Y,sigma, nu=np.inf):
  X2=np.sum(X**2,1).reshape((-1,1))
  XY=X.dot(Y.T)
  Y2=np.sum(Y**2,1).reshape((-1,1))
  D2=X2-2*XY+Y2.T
  D=np.sqrt(D2+1e-10)
  if nu==0.5:      #Laplace
    return np.exp(-D/sigma)
  elif nu==1.5:
    return (1+np.sqrt(3)*D/sigma)*np.exp(-np.sqrt(3)*D/sigma)
  elif nu==2.5:
    return (1+np.sqrt(5)*D/sigma+5*D2/(3*sigma**2))*np.exp(-np.sqrt(5)*D/sigma)
  elif nu==10:     #Cauchy (could have been any number, but I chose 10 for no particular reason)
    return 1/(1+D2/sigma**2)
  else:            #Gaussian
    return np.exp(-0.5*D2/sigma**2)

