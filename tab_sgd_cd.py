import numpy as np
import pickle
from scipy.stats import wilcoxon
import os,sys

data='syn_sgd'
for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

if data[:2]=='bs':
  seeds=range(1,367)
elif data[:3]=='syn':
  seeds=range(101)

if data[-2:]=='cd':
  metrics=['r2','time','sparsity']
  alg_titles=['KCD','K$\\ell_1$R','KGD','KRR']
elif data[-3:]=='sgd':
  metrics=['r2','time']
  alg_titles=['KSGD','K$\\ell_\\infty$R','KGD','KRR']

algs=['kxd','kxr','kgd','krr']
nus = [0.5, 1.5, 2.5, 100, 10]
nu_titles=['\\makecell{$\\nu=1/2$\\\\(Laplace)}','$\\nu=3/2$','$\\nu=5/2$','\\makecell{$\\nu=\infty$\\\\(Gaussian)}', 'Cauchy']
FOLD=0

tab_dict={}
for nu in nus:
  tab_dict[nu]={}
  for alg in algs:
    tab_dict[nu][alg]={}
    for metric in metrics:
      tab_dict[nu][alg][metric]={}

for nu in nus:
  temp_dict={}
  for alg in algs:
    temp_dict[alg]={}
    for metric in metrics:
      temp_dict[alg][metric]=[]
  for seed in seeds:
    if os.path.exists('sgd_cd_data/'+data+'_nu_seed_'+str(nu)+'_'+str(seed)+'.pkl'):
      fi=open('sgd_cd_data/'+data+'_nu_seed_'+str(nu)+'_'+str(seed)+'.pkl','rb')
      data_dict_seed=pickle.load(fi)
      fi.close()
      for alg in algs:
        for metric in metrics:
          temp_dict[alg][metric].append(data_dict_seed[metric][alg])
  for alg in algs:
    for metric in metrics:
      fold0=list(map(lambda v: v[FOLD], temp_dict[alg][metric]))
      med=np.nanmedian(fold0)
      q1=np.nanquantile(fold0,0.25)
      q3=np.nanquantile(fold0,0.75)
      if alg==algs[0]:
        tab_dict[nu][alg][metric]['p_val']= '$-$'
        p_val=1
      else:
        alt= 'greater' if metric=='r2' else 'less'
        fold0_kxd=list(map(lambda v: v[FOLD], temp_dict[algs[0]][metric]))
        p_val=wilcoxon(fold0_kxd, fold0, alternative=alt, nan_policy='omit')[1]
        p_str= f'{p_val:.3g}'
        p_str=p_str.replace('e','\\cdot 10^{').replace('{-0','{-')
        if 'cdot' in p_str: p_str+='}'
        tab_dict[nu][alg][metric]['p_val']= '$'+p_str+'$'
      if p_val<0.01:
        tab_dict[nu][alg][metric]['medq']= '$\\bm{'+f'{med:.2f},\\ ({q1:.2f}, {q3:.2f})'+'}$'
      else:
        tab_dict[nu][alg][metric]['medq']= f'${med:.2f},\\ ({q1:.2f}, {q3:.2f})$'




for nu,nu_title in zip(nus,nu_titles):
  for alg,alg_title in zip(algs,alg_titles):
    nu_str='\\multirow{4}{*}{'+nu_title+'}' if alg==algs[0] else ''
    print_str= nu_str.ljust(52,' ')+' & ' + alg_title.ljust(17,' ')
    for metric in metrics:
      print_str+=' & ' + tab_dict[nu][alg][metric]['medq'].ljust(35,' ')# + ' & ' + tab_dict[nu][alg][metric]['p_val'].ljust(20,' ')
    print(print_str+' \\\\')
  print('\\hline')
