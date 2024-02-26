import numpy as np
from gd_algs import prox_grad
from matplotlib import pyplot as plt
from help_fcts import krr, kern
from matplotlib.lines import Line2D

def mse(y,y_hat):
  return np.mean(np.square(y-y_hat))

def kgd(x_val, x_tr, y_tr, sigma, nu, alg, n_iters=None, y_val=None, step_size=1e-4, t_max=1e3, auto=False):
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
  

def cv_10_kgd(x,y,sigma_bounds,nu,alg,seed, n_sigmas=30):
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
      mse, n_iter=kgd(x_val, x_tr, y_tr, sigma, nu, alg, y_val=y_val, auto=True)
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


def make_data_synth(seed=None, alg='sgd'):
  if not seed is None:
    np.random.seed(seed)
  N_TR=100
  N_TE=1000
  X_MAX=10
  if alg=='sgd':
    def fy(x):
      return np.sin(1/4*2*np.pi*x)
  else:
    def fy(x):
      return np.exp(-5*x**2)
  x_tr=np.random.uniform(-X_MAX,X_MAX,N_TR).reshape((-1,1))
  if alg=='sgd':
    y_tr=fy(x_tr)+0.1*np.random.standard_cauchy((N_TR,1))
    sigma_bounds=[0.1,10]
  else:
    x_tr[0]=0
    y_tr=fy(x_tr)+np.random.normal(0,0.1,(N_TR,1))
    sigma_bounds=[0.1,1]
  x_te=np.linspace(-X_MAX, X_MAX, N_TE).reshape((-1,1))
  y_te=fy(x_te)
  lbda_bounds=[1e-5,10]
  return x_tr, y_tr, x_te, y_te, lbda_bounds, sigma_bounds



N_LS=50

nu=100

lines=[Line2D([0],[0],color='C7',lw=6),plt.plot(0,0,'ok')[0]]
plt.cla()
for ls,c in zip(['-','--','-','--'],['C2','C1','C6','C4']):
  lines.append(Line2D([0],[0],color=c,ls=ls,lw=2))

labs=['True Function', 'Observed Data', 'KSGD/KCD', 'K$\\ell_\\infty$R/K$\\ell_1$R', 'KGD', 'KRR']

fig,axs=plt.subplots(2,1,figsize=(10,5))
#47
#35

for ax,alg, nrm,seed, title in zip(axs,['sgd','cd'],['linf','l1'], [47,35], ['K$\\ell_\\infty$R and KSGD','K$\\ell_1$R and KCD']):
  X_tr, y_tr, X_te, y_te, lbda_bounds, sigma_bounds = make_data_synth(seed,alg)
  
  n_iters_kxd, sigma_kxd=cv_10_kgd(X_tr,y_tr,sigma_bounds,nu,alg,seed, n_sigmas=N_LS)
  fh_kxd, alphah_kxd =kgd(X_te,X_tr, y_tr, sigma_kxd, nu, alg, n_iters_kxd)
  
  n_iters_kgd, sigma_kgd=cv_10_kgd(X_tr,y_tr,sigma_bounds,nu,'gd',seed, n_sigmas=N_LS)
  fh_kgd, alphah_kgd =kgd(X_te,X_tr, y_tr, sigma_kgd, nu, 'gd', n_iters_kgd)
  
  lbda_kxr, sigma_kxr=cv_10_kxrr(X_tr,y_tr,lbda_bounds,sigma_bounds,nu,seed,nrm, n_lbdas=N_LS, n_sigmas=N_LS)
  fh_kxr, alphah_kxr=kxr(X_te, X_tr, y_tr, lbda_kxr,sigma_kxr,nu,nrm)
  
  lbda_krr, sigma_krr=cv_10_kxrr(X_tr,y_tr,lbda_bounds,sigma_bounds,nu,seed, 'ridge', n_lbdas=N_LS, n_sigmas=N_LS)
  fh_krr=krr(X_te,X_tr,y_tr,lbda_krr,sigma_krr,nu, center=False)
  
  ax.cla()
  ax.plot(X_te,y_te,'C7',lw=8)
  ax.plot(X_tr,y_tr,'ok')
  ax.plot(X_te,fh_kxd,'C2',lw=4)
  ax.plot(X_te,fh_kxr,'C1--',lw=3.5)
  ax.plot(X_te,fh_kgd,'C6-',lw=3)
  ax.plot(X_te,fh_krr,'C4--',lw=2.5)
  ax.set_title(title)
  
  fig.legend(lines, labs, loc='lower center', ncol=len(labs))
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.12)
  fig.savefig('figures/syn_sgd_cd_expl.pdf')

