import sys
import os

from run_optimization import opt_als, opt_baseline, opt_naive_bayes, main
from skopt import dummy_minimize, gp_minimize, forest_minimize

optimizer = gp_minimize
args_list = [(opt_naive_bayes, [(0.1,4)]), (opt_als, [(0.001,1), (15,), (30,100)]), 
             (opt_baseline, [(0.1,4), (0.001,1), (15,), (30,100), (0.02,0.10)])]

kwargs = {"verbose": True, "random_state": 0, "n_calls": 120}
opts_list = [optimizer for args in args_list]
kwargs_list = [kwargs for args in args_list]

all_args = list(zip(opts_list, args_list, kwargs_list))

for combination in all_args:
    try:
        main(combination)
    except Exception as e:
        print((str(type(e)) + ": " + str(e)).replace('\n', ' '))
#os.system("shutdown now -h")