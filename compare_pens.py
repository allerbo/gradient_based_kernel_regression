import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from gd_algs import gd_alg, prox_grad
from scipy.linalg import expm

def kern_gauss(X1,X2,sigma):
  if X1.shape[1]==1 and X2.shape[1]==1:
    return np.exp(-0.5*np.square((X1-X2.T)/sigma))
  X1X1=np.sum(np.square(X1),1).reshape((-1,1))
  X2X2=np.sum(np.square(X2),1).reshape((-1,1))
  X1X2=X1@X2.T
  D2=X1X1-2*X1X2+X2X2.T
  return np.exp(-0.5*D2/sigma**2)

def pen_reg_alphah(K, K1,y, lbda, nrm):
    prox_obj=prox_grad(K,y,lbda,nrm,'par')
    alphah_old=np.ones(K.shape[0])
    for ii in range(100000):
      prox_obj.prox_step()
      alphah=prox_obj.get_var()
      if np.allclose(alphah_old, alphah, rtol=0.0001, atol=0.0001):
        break
      alphah_old=np.copy(alphah)
    return alphah

def pen_reg_fh(K, K1,y, lbda, nrm):
    prox_obj=prox_grad(np.vstack((K,K1)),y,lbda,nrm,'pred')
    fh_old=np.ones(K.shape[0])
    for ii in range(100000):
      prox_obj.prox_step()
      fh=prox_obj.get_var()
      if np.allclose(fh, fh_old, rtol=0.0001, atol=0.0001):
        break
      fh_old=np.copy(fh)
    return fh


def gd_reg(K, K1, y, alg, fh_lbda):
    mse_max=100
    step_size=0.0001
    gd_obj=gd_alg(K,y,alg,'par',step_size)
    best_mse=np.inf
    best_alphah=np.zeros((K.shape[0],1))
    alphah=best_alphah
    mse_counter=0
    while 1:
      gd_obj.gd_step()
      alphah=gd_obj.get_alpha()
      fh= K1@alphah
      mse=np.mean((fh-fh_lbda)**2)
      mse_counter+=1
      if mse<best_mse:
        best_mse=mse
        best_alphah=alphah
        mse_counter=0
      if mse_counter>mse_max:
        break
    return best_alphah

def f(x):
  y=np.sin(np.pi*x)
  return y

N=1001
l=3
seed=0

np.random.seed(seed)
#x=np.sort(np.random.uniform(0,l,n).reshape((-1,1)),0)
x=np.array([0.1,0.5,1.0,1.5,2.1,2.5,2.9]).reshape((-1,1))
n=x.shape[0]
y=f(x)+np.random.normal(0,.01,x.shape)
y[1]+=3
x1=np.linspace(0,l,N).reshape((-1,1))
sigma=0.3
K=kern_gauss(x,x,sigma)
K1=kern_gauss(x1,x,sigma)
alphah0=np.linalg.solve(K,y)
fh0=K1@alphah0

labs1=['Observed Data', 'Non- and Fully Reguralized Solutions','Gradient-Based Optimization', 'Explicit Regularization']
lines1=[plt.plot(0,0,'ok')[0]]
plt.cla()
for c in ['C7','C2','C1']:
  lines1.append(Line2D([0],[0],color=c,lw=2))

labs2=['Non-Reguralized Solution','Gradient-Based Optimization$', 'Explicit Regularization$']
lines2=[]
for c in ['C7','C2','C1']:
  lines2.append(Line2D([0],[0],color=c,lw=4))

labs3=['Observed Data', 'Non- and Fully Reguralized Solutions', 'Explicit Regularization']
lines3=[plt.plot(0,0,'ok')[0]]
plt.cla()
for c in ['C7','C2']:
  lines3.append(Line2D([0],[0],color=c,lw=2))


fig1,axss1=plt.subplots(2,3,figsize=(10,5))
fig2,axss2=plt.subplots(2,3,figsize=(10,5))
fig3,axss3=plt.subplots(2,3,figsize=(10,5))

for axss in [axss1,axss2]:
  axss[0,0].set_title('KRR and KGF',fontsize=13)
  axss[0,1].set_title('K$\\ell_1$R and KCD',fontsize=13)
  axss[0,2].set_title('K$\\ell_\\infty$R and KSGD',fontsize=13)

axss3[0,0].set_title('KRR',fontsize=13)
axss3[0,1].set_title('K$\\ell_1$R',fontsize=13)
axss3[0,2].set_title('K$\\ell_\\infty$R',fontsize=13)

axss1[0,0].set_ylabel('$\\hat{f}$',fontsize=13)
axss1[1,0].set_ylabel('$\\hat{f}$',fontsize=13)
axss2[0,0].set_ylabel('$\\alpha$',fontsize=13)
axss2[1,0].set_ylabel('$\\alpha$',fontsize=13)
axss3[0,0].set_ylabel('$\\hat{f}$',fontsize=13)
axss3[1,0].set_ylabel('$\\hat{f}$',fontsize=13)

lbdass_a=[[.0007,0.0015,0.003],[.0002,0.0007,0.0007]]
lbdass_f=[[.00065,0.0021,.65],[.0002,0.001,.095]]

BW=0.35
LW=2
fig1.legend(lines1, labs1, loc='lower center', ncol=3)
fig1.tight_layout()
fig1.subplots_adjust(bottom=0.17)
fig2.legend(lines2, labs2, loc='lower center', ncol=3)
fig2.tight_layout()
fig2.subplots_adjust(bottom=0.13)
fig3.legend(lines3, labs3, loc='lower center', ncol=3)
fig3.tight_layout()
fig3.subplots_adjust(bottom=0.17)

lbdass_a=[[1,0.0015,0.003],[.35,0.0007,0.0007]]

for axs1,axs2,axs3,lbdas_a,lbdas_f in zip(axss1,axss2,axss3,lbdass_a,lbdass_f):
  for ax1, ax2, ax3, nrm, alg,lbda_a,lbda_f in zip(axs1,axs2,axs3,['l2','l1','linf'], ['gd','cd','sgd'],lbdas_a,lbdas_f):
    if nrm=='l2':
      alphah_l=np.linalg.inv(K+lbda_a*np.eye(n))@y
      alphah_t=np.linalg.inv(K)@(np.eye(n)-expm(-1/lbda_a*K))@y
      fh_l=K1@alphah_l
    else:
      alphah_l= pen_reg_alphah(K,K1,y, lbda_a, nrm)
      fh_l=K1@alphah_l
      alphah_t=gd_reg(K,K1,y,alg, fh_l)
    fh_t=K1@alphah_t
    if nrm=='l2':
      alphah_t2=np.linalg.inv(K)@(np.eye(n)-expm(-1/lbda_a*K))@y
      fh_t2=K1@alphah_t2
    fh_l_f= pen_reg_fh(K,K1,y, lbda_f, nrm)
    
    ax1.plot(x1,fh0,'C7', lw=4)
    ax1.plot(x1,np.zeros(x1.shape),'C7', lw=4)
    ax1.plot(x,y,'ok')
    ax1.plot(x1,fh_l,'C1',lw=3.5)
    ax1.plot(x1,fh_t,'C2',lw=2)
    fig1.savefig('figures/compare_pen_af.pdf')
    
    ax2.bar(1+np.arange(n),np.squeeze(alphah0),color='C7', width=2*BW)
    ax2.bar(1+np.arange(n)-BW/2,np.squeeze(alphah_t),color='C2', width=BW)
    ax2.bar(1+np.arange(n)+BW/2,np.squeeze(alphah_l),color='C1', width=BW)
    ax2.set_xticks(1+np.arange(n))
    fig2.savefig('figures/compare_pen_a.pdf')
    
    ax3.plot(x1,fh0,'C7', lw=4)
    ax3.plot(x1,np.zeros(x1.shape),'C7', lw=4)
    ax3.plot(x,y,'ok')
    ax3.plot(x1,fh_l_f[n:],'C2',lw=3.5)
    fig3.savefig('figures/compare_pen_f.pdf')

