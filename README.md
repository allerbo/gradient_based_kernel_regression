This is the code used in the article **Solving Kernel Ridge Regression with Gradient-Based Optimization Methods**, available at http://arxiv.org/abs/2306.16838.


## Figures 1, 2 and 4:
```
python compare_pens.py
```

## Figure 3:
```
python syn_sgd_cd_expl.py
```

## Table 3:
```
bash run_sgd_cd.py syn_cd             #Calls sgd_cd.py.
python tab_sgd_cd.py data=\"syn_cd\"  #Uses data in sgd_cd_data
```

## Table 4:
```
bash run_sgd_cd.py syn_sgd            #Calls sgd_cd.py.
python tab_sgd_cd.py data=\"syn_sgd\" #Uses data in sgd_cd_data
```

## Table 5:
```
bash run_sgd_cd.py bs_cd              #Calls sgd_cd.py.
python tab_sgd_cd.py data=\"bs_cd\"   #Uses data in sgd_cd_data
```

## Table 6:
```
bash run_sgd_cd.py bs_sgd             #Calls sgd_cd.py.
python tab_sgd_cd.py data=\"bs_sgd\"  #Uses data in sgd_cd_data
```
