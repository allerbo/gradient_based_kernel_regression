import numpy as np
import pickle
from scipy.stats import wilcoxon
import os,sys

kern=3
suf=''
aim='cd'
for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

seeds=range(101)

if aim=='cd':
  metrics=['r2','time','sparsity']
  algs=['kxd','kxr','kgd','krr','kegd']
  alg_titles=['KCD','K$\\ell_1$R','KGD','KRR','\\textit{KEGD}']
elif aim=='sgd':
  metrics=['r2','time']
  algs=['kxd','kxr','kgd','krr']
  alg_titles=['KSGD','K$\\ell_\\infty$R','KGD','KRR']

datas=['synth', 'wood','house','casp','bs']
data_titles=['Synthetic', '\\makecell{Aspen\\\\Fibres}','\\makecell{California\\\\Housing}','\\makecell{Protein\\\\Structure}','\\makecell{U.K.\\ Black\\\\Smoke}']


nus = [100, 0.5, 1.5, 2.5, 10]
nu_titles=['\\makecell{Matérn,\\\\$\\nu=\\infty$\\\\(Gaussian)}', '\\makecell{Matérn,\\\\$\\nu=1/2$\\\\(Laplace)}','\\makecell{Matérn,\\\\$\\nu=3/2$}','\\makecell{Matérn,\\\\$\\nu=5/2$}', 'Cauchy']

tab_dict={}
for nu in nus:
  tab_dict[nu]={}
  for data in datas:
    tab_dict[nu][data]={}
    for alg in algs:
      tab_dict[nu][data][alg]={}
      for metric in metrics:
        tab_dict[nu][data][alg][metric]={}


for nu in nus:
  temp_dict={}
  for data in datas:
    temp_dict[data]={}
    for alg in algs:
      temp_dict[data][alg]={}
      for metric in metrics:
        temp_dict[data][alg][metric]=[]
    for seed in seeds:
      if os.path.exists(aim+'_data/'+data+'_'+aim+'_'+str(nu)+'_'+str(seed)+suf+'.pkl'):
        fi=open(aim+'_data/'+data+'_'+aim+'_'+str(nu)+'_'+str(seed)+suf+'.pkl','rb')
        data_dict_seed=pickle.load(fi)
        fi.close()
        for alg in algs:
          for metric in metrics:
            temp_dict[data][alg][metric].append(data_dict_seed[metric][alg])
    for alg in algs:
      for metric in metrics:
        med=np.nanmedian(temp_dict[data][alg][metric])
        q1=np.nanquantile(temp_dict[data][alg][metric],0.25)
        q3=np.nanquantile(temp_dict[data][alg][metric],0.75)
        if alg==algs[0]:
          tab_dict[nu][data][alg][metric]['p_val']= '$-$'
          p_val=1
        else:
          alt= 'greater' if metric=='r2' else 'less'
          p_val=wilcoxon(temp_dict[data]['kxd'][metric], temp_dict[data][alg][metric], alternative=alt, nan_policy='omit')[1]
          p_str= f'{p_val:.3g}'
          p_str=p_str.replace('e','\\cdot 10^{').replace('{-0','{-')
          if 'cdot' in p_str: p_str+='}'
          tab_dict[nu][data][alg][metric]['p_val']= '$'+p_str+'$'
        if alg=='kegd':
          if metric=='time':
            tab_dict[nu][data][alg][metric]['medq']= '$\\mathit{'+f'{round(med)},\\ ({round(q1)}, {round(q3)})'+'}$'
          else:
            tab_dict[nu][data][alg][metric]['medq']= '$\\mathit{'+f'{med:.2f},\\ ({q1:.2f}, {q3:.2f})'+'}$'
        elif p_val<0.05:
          if metric=='time':
            tab_dict[nu][data][alg][metric]['medq']= '$\\bm{'+f'{round(med)},\\ ({round(q1)}, {round(q3)})'+'}$'
          else:
            tab_dict[nu][data][alg][metric]['medq']= '$\\bm{'+f'{med:.2f},\\ ({q1:.2f}, {q3:.2f})'+'}$'
        else:
          if metric=='time':
            tab_dict[nu][data][alg][metric]['medq']= f'${round(med)},\\ ({round(q1)}, {round(q3)})$'
          else:
            tab_dict[nu][data][alg][metric]['medq']= f'${med:.2f},\\ ({q1:.2f}, {q3:.2f})$'



print('')
for nu, nu_title in zip(nus, nu_titles):
  for data,data_title in zip(datas,data_titles):
    data_str='\\multirow{20}{*}{'+nu_title+'}' if data==datas[0] else ''
    for alg,alg_title in zip(algs,alg_titles):
      alg_str=data_str+' & \\multirow{4}{*}{'+data_title+'}' if alg==algs[0] else ' &'
      print_str= alg_str.ljust(72,' ')+' & ' + alg_title.ljust(17,' ')
      for metric in metrics:
        print_str+=' & ' + tab_dict[nu][data][alg][metric]['medq'].ljust(35,' ')
      print(print_str+' \\\\')
    if data==datas[-1]:
      print('\\hline')
    elif aim=='cd':
      print('\\cline{2-6}')
    else:
      print('\\cline{2-5}')
