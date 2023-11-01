import numpy as np
import pandas as pd
from help_fcts import r2, krr, kern
from gd_algs import prox_grad
import pickle
import time
import sys

def mse(y,y_hat):
  return np.mean(np.square(y-y_hat))

def kgd(x_val, x_tr, y_tr, sigma, nu, alg, n_iters=None, y_val=None, step_size=0.01, t_max=1e3, auto=False):
  K_tr=kern(x_tr, x_tr, sigma, nu)
  K_val=kern(x_val, x_tr, sigma, nu)
  
  alpha=np.zeros(y_tr.shape)
  if auto:
    best_mse=np.inf
    mse_counter=0
    n_iters=int(t_max/step_size)
  for n_iter in range(n_iters):
    grad=K_tr@alpha-y_tr
    if alg=='gd':
      alpha-=step_size*grad
    elif alg=='sgd':
      alpha-=step_size*np.sign(grad)
    elif alg=='cd':
      alpha-=step_size*np.sign(grad)*(np.abs(grad)==np.max(np.abs(grad)))
    
    if auto:
      mse_val=mse(y_val,K_val@alpha)
      if mse_val<best_mse:
        best_mse=mse_val
        best_n_iter=n_iter
        mse_counter=0
      mse_counter+=1
      if mse_counter>100:
        break
  if auto:
    return best_mse, best_n_iter
  return K_val@alpha, alpha
  

def cv_10_kgd(x,y,sigma_bounds,nu,alg,seed, step_size=0.01, n_sigmas=30):
  sigmas=np.geomspace(*sigma_bounds,n_sigmas)
  n=x.shape[0]
  np.random.seed(seed)
  per=np.random.permutation(n)
  folds=np.array_split(per,10)
  best_params=(np.inf,None,None)
  for sigma in sigmas:
    mses=[]
    n_iters=[]
    for v_fold in range(len(folds)):
      t_folds=np.concatenate([folds[t_fold] for t_fold in range(len(folds)) if v_fold != t_fold])
      v_folds=folds[v_fold]
      x_tr=x[t_folds,:]
      y_tr=y[t_folds,:]
      x_val=x[v_folds,:]
      y_val=y[v_folds,:]
      mse, n_iter=kgd(x_val, x_tr, y_tr, sigma, nu, alg, y_val=y_val, step_size=step_size, auto=True)
      mses.append(mse)
      n_iters.append(n_iter)
    mean_mse=np.mean(mses)
    if mean_mse<best_params[0]:
      best_params=(mean_mse, int(np.mean(n_iters)),sigma)
  return best_params[1], best_params[2]

def kxr(x1,x_tr,y_tr,lbda,sigma, nu, nrm, t_max=1000):
  K=kern(x_tr,x_tr,sigma,nu)
  
  prox_obj=prox_grad(K,y_tr,lbda,nrm,'par')
  alphah_old=np.ones(K.shape[0])
  for ii in range(t_max):
    prox_obj.prox_step()
    alphah=prox_obj.get_var()
    if np.allclose(alphah_old, alphah, rtol=0.0001, atol=0.0001):
      break
    alphah_old=np.copy(alphah)
  return kern(x1,x_tr,sigma, nu)@alphah, alphah

def cv_10_kxrr(x,y,lbda_bounds,sigma_bounds,nu,seed,nrm,n_lbdas=30,n_sigmas=30):
  sigmas=np.geomspace(*sigma_bounds,n_sigmas)
  lbdas=np.geomspace(*lbda_bounds,n_lbdas)
  n=x.shape[0]
  np.random.seed(seed)
  per=np.random.permutation(n)
  folds=np.array_split(per,10)
  best_params=(np.inf,None,None)
  for lbda in lbdas:
    for sigma in sigmas:
      mses=[]
      n_iters=[]
      for v_fold in range(len(folds)):
        t_folds=np.concatenate([folds[t_fold] for t_fold in range(len(folds)) if v_fold != t_fold])
        v_folds=folds[v_fold]
        x_tr=x[t_folds,:]
        y_tr=y[t_folds,:]
        x_val=x[v_folds,:]
        y_val=y[v_folds,:]
        if nrm=='ridge':
          fh=krr(x_val, x_tr, y_tr, lbda,sigma,nu,center=False)
        else:
          fh=kxr(x_val, x_tr, y_tr, lbda,sigma,nu,nrm)[0]
        mses.append(mse(y_val,fh))
      mean_mse=np.mean(mses)
      if mean_mse<best_params[0]:
        best_params=(mean_mse, lbda, sigma)
  return best_params[1], best_params[2]



def in_hull(p, hull):
  from scipy.spatial import Delaunay
  if not isinstance(hull,Delaunay):
    hull = Delaunay(hull)
  
  return hull.find_simplex(p)>=0

def make_data_sin(seed=None):
  if not seed is None:
    np.random.seed(seed)
  X_MAX=10
  def fy(x):
    return np.sin(1/4*2*np.pi*x)
  x_tr=np.random.uniform(-X_MAX,X_MAX,N_TR).reshape((-1,1))
  y_tr=fy(x_tr)+0.1*np.random.standard_cauchy((N_TR,1))
  x1=np.linspace(-X_MAX, X_MAX, N_TE).reshape((-1,1))
  y1=fy(x1)
  lbda_bounds=[1e-5,10]
  sigma_bounds=[0.1,10]
  return x_tr, y_tr, x1, y1, lbda_bounds, sigma_bounds


N_TR=100
N_TE=1000

def make_data_gauss(seed=None):
  if not seed is None:
    np.random.seed(seed)
  X_MAX=10
  def fy(x):
    return np.exp(-5*x**2)
  x_tr=np.random.uniform(-X_MAX,X_MAX,N_TR).reshape((-1,1))
  x_tr[0]=0
  y_tr=fy(x_tr)+np.random.normal(0,0.1,(N_TR,1))
  x1=np.linspace(-X_MAX, X_MAX, N_TE).reshape((-1,1))
  y1=fy(x1)
  lbda_bounds=[1e-5,10]
  sigma_bounds=[0.1,1]
  return x_tr, y_tr, x1, y1, lbda_bounds, sigma_bounds


def make_data_bs(seed, alg):
  bs_data=pd.read_csv('bs_2000.csv',sep=',').to_numpy()
  np.random.seed(0)
  bs_data1=bs_data[bs_data[:,1]==seed]
  np.random.shuffle(bs_data1)
  X=bs_data1[:,8:10]
  X=(X-np.mean(X, 0))/np.std(X,0)
  y=bs_data1[:,5].reshape((-1,1))
  y=y-np.mean(y)
  if alg=='sgd':
    y*=(1+0.01*np.abs(np.random.standard_cauchy(y.shape)))
  n=X.shape[0]
  per=np.random.permutation(n)
  folds=np.array_split(per,10)
  sigma_bounds=[0.01,10]
  lbda_bounds=[1e-3,1]
  return X, y, folds, lbda_bounds, sigma_bounds

def cv_split(X, y, v_fold):
  t_idxs=np.concatenate([folds[t_fold] for t_fold in range(len(folds)) if v_fold != t_fold])
  v_idxs=folds[v_fold]
   
  X_tr=X[t_idxs,:]
  X_te=X[v_idxs,:]
  y_tr=y[t_idxs,:]
  y_te=y[v_idxs,:]
  
  #Remove outside convex hull
  in_ch=in_hull(X_te,X_tr)
  X_te=X_te[in_ch,:]
  y_te=y_te[in_ch,:]
  
  return X_tr, y_tr, X_te, y_te

N_LS=30

nu=100
seed=1
data='syn_cd'
for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if data=='syn_sgd':
  alg='sgd'
  nrm='linf'
  make_data=make_data_sin
elif data=='syn_cd':
  alg='cd'
  nrm='l1'
  make_data=make_data_gauss
elif data=='bs_cd':
  alg='cd'
  nrm='l1'
  make_data=make_data_bs
elif data=='bs_sgd':
  alg='sgd'
  nrm='linf'
  make_data=make_data_bs


metrics=['r2','time','sparsity']
algs=['kxd', 'kgd','kxr','krr']
data_dict={}
for metric in metrics:
  data_dict[metric]={}
  for alg1 in algs:
    data_dict[metric][alg1]=[]

if data=='bs_cd':
  for metric in metrics:
    data_dict[metric]['kxd1']=[]

if data[:2]=='bs':
  X, y, folds, lbda_bounds, sigma_bounds = make_data_bs(seed, alg)
else:
  X_tr, y_tr, X_te, y_te, lbda_bounds, sigma_bounds=make_data(seed)
  folds=[0]


for v_fold in range(len(folds)):
  if data[:2]=='bs':
    X_tr, y_tr, X_te, y_te = cv_split(X, y, v_fold)
  n_tr=X_tr.shape[0]

  t1=time.time()
  n_iters_kxd, sigma_kxd=cv_10_kgd(X_tr,y_tr,sigma_bounds,nu,alg,seed, n_sigmas=N_LS)
  fh_kxd, alphah_kxd =kgd(X_te,X_tr, y_tr, sigma_kxd, nu, alg, n_iters_kxd)
  data_dict['time']['kxd'].append(time.time()-t1)

  t1=time.time()
  n_iters_kgd, sigma_kgd=cv_10_kgd(X_tr,y_tr,sigma_bounds,nu,'gd',seed, n_sigmas=N_LS)
  fh_kgd, alphah_kgd =kgd(X_te,X_tr, y_tr, sigma_kgd, nu, 'gd', n_iters_kgd)
  data_dict['time']['kgd'].append(time.time()-t1)
  
  t1=time.time()
  lbda_kxr, sigma_kxr=cv_10_kxrr(X_tr,y_tr,lbda_bounds,sigma_bounds,nu,seed,nrm, n_lbdas=N_LS, n_sigmas=N_LS)
  fh_kxr, alphah_kxr=kxr(X_te, X_tr, y_tr, lbda_kxr,sigma_kxr,nu,nrm)
  data_dict['time']['kxr'].append(time.time()-t1)
  
  t1=time.time()
  lbda_krr, sigma_krr=cv_10_kxrr(X_tr,y_tr,lbda_bounds,sigma_bounds,nu,seed, 'ridge', n_lbdas=N_LS, n_sigmas=N_LS)
  fh_krr=krr(X_te,X_tr,y_tr,lbda_krr,sigma_krr,nu, center=False)
  data_dict['time']['krr'].append(time.time()-t1)
      
  data_dict['r2']['kxd'].append(r2(y_te,fh_kxd))
  data_dict['r2']['kgd'].append(r2(y_te,fh_kgd))
  data_dict['r2']['kxr'].append(r2(y_te,fh_kxr))
  data_dict['r2']['krr'].append(r2(y_te,fh_krr))
  
  data_dict['sparsity']['kxd'].append(np.sum(alphah_kxd!=0)/n_tr)
  data_dict['sparsity']['kgd'].append(np.sum(alphah_kgd!=0)/n_tr)
  data_dict['sparsity']['kxr'].append(np.sum(alphah_kxr!=0)/n_tr)
  data_dict['sparsity']['krr'].append(1)

  if data=='bs_cd':
    t1=time.time()
    n_iters_kxd1, sigma_kxd1=cv_10_kgd(X_tr,y_tr,sigma_bounds,nu,alg,seed, step_size=0.1, n_sigmas=N_LS)
    fh_kxd1, alphah_kxd1 =kgd(X_te,X_tr, y_tr, sigma_kxd1, nu, alg, n_iters_kxd1, step_size=0.1)
    data_dict['time']['kxd1'].append(time.time()-t1)
    data_dict['r2']['kxd1'].append(r2(y_te,fh_kxd1))
    data_dict['sparsity']['kxd1'].append(np.sum(alphah_kxd1!=0)/n_tr)


fi=open('sgd_cd_data/'+data+'_nu_seed_'+str(nu)+'_'+str(seed)+'.pkl','wb')
pickle.dump(data_dict,fi)
fi.close()
