import numpy as np
from time import sleep

class gd_alg():
  def __init__(self, K, y, alg, space='par', lr=0.01, gamma=0, alpha_egd=0.5, var0=None):
    assert space in ['par', 'pred', 'orac'], 'Non-valid space!'
    assert alg in ['cd', 'gd', 'egd', 'sgd', 'esgd', 'adam'], 'Non-valid alg!'
    self.K=K
    self.y=y
    self.alg=alg
    self.space=space
    self.lr=lr
    self.gamma=gamma
    self.alpha_egd=alpha_egd
    n=y.shape[0]
    if space=='par':
      if var0 is None:
        self.var=np.zeros((n,1))
      else:
        self.var=var0
      self.m=np.zeros((n,1))
      self.v=np.zeros((n,1))
    elif space=='pred' or space=='orac':
      ns=K.shape[0]
      if var0 is None:
        self.var=np.zeros((ns,1))
      else:
        self.var=var0
      self.Ih=np.hstack((np.eye(n),np.zeros((n,ns-n))))
      self.m=np.zeros((ns,1))
      self.v=np.zeros((ns,1))
    self.var_old=np.copy(self.var)

    self.b1=0.9
    self.b2=0.999
    self.eps=1e-7
    self.t=1

  def update(self,K,var):
    self.K=K
    self.var=var

  def gd_step(self):
    if self.space=='par':
      grad = self.K@self.var-self.y 
    elif self.space=='pred':
      grad = self.K@(self.Ih@self.var-self.y)
    elif self.space=='orac':
      grad = self.K@(self.var-self.y)
    if self.alg=='cd':
      I_cd=(np.abs(grad)==np.max(np.abs(grad)))
      self.var-= self.lr*I_cd*np.sign(grad)
    elif self.alg=='gd':
      self.var-=self.lr*grad
    elif self.alg=='egd':
      I_egd=(np.abs(grad)>=self.alpha_egd*np.max(np.abs(grad)))
      self.var-= self.lr*I_egd*(self.alpha_egd*np.sign(grad)+(1-self.alpha_egd)*grad)
    elif self.alg=='sgd':
      self.var-=self.lr*np.sign(grad)
    elif self.alg=='esgd':
      self.var-= self.lr*(self.alpha_egd*np.sign(grad)+(1-self.alpha_egd)*grad)
    elif self.alg=='adam':
      self.m=self.b1*self.m+(1-self.b1)*grad
      self.v=self.b2*self.v+(1-self.b2)*grad**2
      mh=self.m/(1-self.b1**self.t)
      vh=self.v/(1-self.b2**self.t)
      self.var-=self.lr/(np.sqrt(vh)+self.eps)*mh
      self.t+=1
    else:
      print('Non-valid algorithm!')
  
  def get_var(self):
    return self.var

  def get_fs(self):
    return self.get_var()

  def get_alpha(self):
    return self.get_var()

  def get_grad(self):
    if self.space=='par':
      grad = self.K@self.var-self.y 
    elif self.space=='pred':
      grad = self.K@(self.Ih@self.var-self.y)
    return grad

class prox_grad():
  def __init__(self, K,y,lbda, nrm, space='par', lr=0.001):
    assert space in ['par', 'pred', 'par1'], 'Non-valid space!'
    assert nrm in ['l1','l2','linf'], "Non-valid norm!"
    self.K=K
    self.y=y
    self.lbda=lbda
    self.nrm=nrm
    self.space=space
    self.lr=lr
    n=y.shape[0]
    if space=='par' or space=='par1':
      self.var=np.zeros((n,1))
    elif space=='pred':
      ns=K.shape[0]
      self.var=np.zeros((ns,1))
      self.Ih=np.hstack((np.eye(n),np.zeros((n,ns-n))))

  def prox(self, x):
    if self.nrm=='l1':
      return np.sign(x)*np.maximum(np.abs(x)-self.lbda,0)
    if self.nrm=='l2':
      return x/(1+self.lbda)
    if self.nrm=='linf':
      return x-self.euclidean_proj_l1ball(np.squeeze(x),self.lbda).reshape((-1,1))

  def prox_step(self):
    if self.space=='par1':
      grad=self.K@(self.K@self.var-self.y)
    else:
      grad=self.K@self.var-self.y if self.space=='par' else self.K@(self.Ih@self.var-self.y)
    var_g=self.var-self.lr*grad
    self.var=self.prox(var_g)
  
  def get_var(self):
    return self.var

  def get_fs(self):
    return self.get_var()

  def get_alpha(self):
    return self.get_var()


  #The code below is taken from https://gist.github.com/daien/1272551
  def euclidean_proj_simplex(self, v, s=1):
      """ Compute the Euclidean projection on a positive simplex
      Solves the optimisation problem (using the algorithm from [1]):
          min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
      Parameters
      ----------
      v: (n,) numpy array,
         n-dimensional vector to project
      s: int, optional, default: 1,
         radius of the simplex
      Returns
      -------
      w: (n,) numpy array,
         Euclidean projection of v on the simplex
      Notes
      -----
      The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
      Better alternatives exist for high-dimensional sparse vectors (cf. [1])
      However, this implementation still easily scales to millions of dimensions.
      References
      ----------
      [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
          John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
          International Conference on Machine Learning (ICML 2008)
          http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
      """
      assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
      n, = v.shape  # will raise ValueError if v is not 1-D
      # check if we are already on the simplex
      if v.sum() == s and np.alltrue(v >= 0):
          # best projection: itself!
          return v
      # get the array of cumulative sums of a sorted (decreasing) copy of v
      u = np.sort(v)[::-1]
      cssv = np.cumsum(u)
      # get the number of > 0 components of the optimal solution
      rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
      # compute the Lagrange multiplier associated to the simplex constraint
      theta = float(cssv[rho] - s) / (rho+1)
      # compute the projection by thresholding v using theta
      w = (v - theta).clip(min=0)
      return w
  
  
  def euclidean_proj_l1ball(self, v, s=1):
      """ Compute the Euclidean projection on a L1-ball
      Solves the optimisation problem (using the algorithm from [1]):
          min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
      Parameters
      ----------
      v: (n,) numpy array,
         n-dimensional vector to project
      s: int, optional, default: 1,
         radius of the L1-ball
      Returns
      -------
      w: (n,) numpy array,
         Euclidean projection of v on the L1-ball of radius s
      Notes
      -----
      Solves the problem by a reduction to the positive simplex case
      See also
      --------
      euclidean_proj_simplex
      """
      assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
      n, = v.shape  # will raise ValueError if v is not 1-D
      # compute the vector of absolute values
      u = np.abs(v)
      # check if v is already a solution
      if u.sum() <= s:
          # L1-norm is <= s
          return v
      # v is not already a solution: optimum lies on the boundary (norm == s)
      # project *u* on the simplex
      w = self.euclidean_proj_simplex(u, s=s)
      # compute the solution to the original problem on v
      w *= np.sign(v)
      return w
  
