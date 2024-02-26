This is the code used in the article **Solving Kernel Ridge Regression with Gradient-Based Optimization Methods**, available at http://arxiv.org/abs/2306.16838.


## Figures 1, 2 and 4:
```
python compare_pens.py
```

## Table 3, 4, 6, 7, 8 and 9:
```
bash run_sgd_cd.py                    #Calls sgd_cd.py.
python tab_sgd_cd.py aim=\"sgd\"      #Uses data in sgd_data
python tab_sgd_cd_all.py aim=\"sgd\"  #Uses data in sgd_data
python tab_sgd_cd.py aim=\"cd\"       #Uses data in cd_data
python tab_sgd_cd_all.py aim=\"cd\"   #Uses data in cd_data
```

## Figure 3:
```
python syn_sgd_cd_expl.py
```

