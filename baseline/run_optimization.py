import sys
sys.path.append('..')

import os
import oyaml as yaml
import lib.utils as utils
import run_script
from skopt import dummy_minimize, gp_minimize, forest_minimize, dump

folder_path = "runs/optimizer_runs/"


def get_unique_filename(name_str):
    """add digit to create unique name, either at last position or at
    4th last position (if .yml).
    """
    def get_formatted_name():
        if ".yml" in name_str:
            return name_str[:-4] + str(i) + name_str[-4:]
        else:
            return name_str + str(i)
    i = 1
    while os.path.exists(folder_path + get_formatted_name()):
        i += 1
    return get_formatted_name()

def write_yml_file(yml_params, file_name=get_unique_filename("run_file.yml")):
    with open(folder_path + file_name, 'w') as w_file:
        yaml.dump(yml_params, w_file)

    return file_name

def run_config(yml_params):
    run_file_name = write_yml_file(yml_params)
    print("Args: " + str(yml_params))
    res_recall = -run_script.main(["run_optimization.py", run_file_name], folder_path, log_output=False)
    os.remove(folder_path + run_file_name)
    return res_recall

def opt_naive_bayes(params=[1]):
    [NB_smoothing] = params

    yml_params = {"--base_model": "NaiveBayesEstimator", "--num_k_folds": "10",
                  "--topk": "3", "--NB_smoothing": str(NB_smoothing)}

    return run_config(yml_params)

def opt_als(params=[0.01, 15, 50]):
    regularization, iterations, factors, confidence = params

    yml_params = {"--base_model": "ALSEstimator", "--num_k_folds": "10", "--topk": "3",
                  "--regularization": str(regularization), "--iterations": str(iterations),
                  "--factors": str(factors), "--confidence": str(confidence)}

    return run_config(yml_params)

def opt_baseline(params=[1, 0.01, 15, 50, 0.06]):
    NB_smoothing, regularization, iterations, factors, content_scale_factor, confidence = params

    yml_params = {"--base_model": "BaselineModel", "--num_k_folds": "10", "--topk": "10",
                  "--regularization": str(regularization), "--iterations": str(iterations),
                  "--factors": str(factors), "--NB_smoothing": str(NB_smoothing),
                  "--content_scale_factor": str(content_scale_factor),
                  "--confidence": str(confidence)}

    return run_config(yml_params)

def main(combinations=None):
    if combinations:
        optimizer, args, kwargs = combinations
    else:
        optimizer = gp_minimize
        args = (opt_naive_bayes, [(0.1,4)])
        kwargs = {"verbose": True, "random_state": 0, "n_calls": 100}
    optim_log_name = get_unique_filename("optim_log")
    sys.stdout = utils.Logger(folder_path + optim_log_name)

    print("optimizer:")
    print(optimizer)
    print("opt-args:")
    print(args)
    print("opt-kwargs:")
    print(kwargs)
    print("")

    res = optimizer(*args, **kwargs)

    print("\n")
    print(res)

    sys.stdout = sys.stdout.terminal

    dump(res, folder_path + "res_files/" + optim_log_name + "_res_dump")

if __name__ == '__main__':
    main(sys.argv[1:])