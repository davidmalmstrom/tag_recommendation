# Trains and tests a configuration. Specify config as argument. Make sure that the configuration has '--test_dataset: "1"'.

import sys
sys.path.append("..")

import nncf.NeuMF as NeuMF
import nncf.MLP as MLP
import nncf.GMF as GMF
from nncf.evaluate_recall import evaluate_model_recall
import oyaml as yaml
import scipy.sparse as sp
import lib.notebook_helpers as nh
import pickle
import nncf.run_script
import os

TEST_PART_SIZE = 2000

def train_model(model_runfile_path, vscode=False):
    """Uses the NeuMF-module to build and train a model. This will modify the .yml file,
    adding the model weights path so that they can be loaded by the test_model functions.
    """
    read_params(model_runfile_path)  # for the assert
    if vscode:
        nncf.run_script.main(["run_model_test.py", model_runfile_path, "vscode"])
    else:
        nncf.run_script.main(["run_model_test.py", model_runfile_path])

def read_params(model_runfile_path):
    """Also asserts that the yml file is valid for testing
    """
    with open(model_runfile_path) as run_file:
        yaml_params = yaml.safe_load(run_file)
        params_mid = {k[2:]: v for k, v in yaml_params.items()}  # yml file keys has leading "--"
        params = {k: eval(v) if k in ['layers', 'reg_layers', 'num_factors', 'topk'] 
                      else v for k, v in params_mid.items()}
        try:
            assert(params['test_dataset'] == '1')
        except (KeyError, AssertionError) as e:
            print(repr(e))
            print("Not a valid test yml file; please specify \"--test_dataset: 1\" in the yml-file.")
            sys.exit(1)
        try:
            assert(params['dataset_name_prepend'])
        except (KeyError, AssertionError) as e:
            params['dataset_name_prepend'] = ""

        return params

def get_test_set(prepend):
    test_set_path = os.path.join(os.path.dirname(__file__), '../data/' + prepend + 'test_tag_dataset.pkl')
    with open(test_set_path, 'rb') as f:
        X, y_test, _, _, _, test_y = pickle.load(f)
        test_set = sp.dok_matrix(y_test)
    return test_set, test_y, X

def build_model(params, data_shape, num_autotags):
    num_users, num_items = data_shape
    try:
        if params['nn_model'] == 'NeuMF':
            reg_mf = 0
            model = NeuMF.get_model(num_users, num_autotags, num_items, params['num_factors'], params['layers'], params['reg_layers'], reg_mf)
        elif params['nn_model'] == "GMF":
            model = GMF.get_model(num_users, num_autotags, num_items, params['num_factors'])
        elif params['nn_model'] == "MLP":
            model = MLP.get_model(num_users, num_autotags, num_items, params['layers'], params['reg_layers'])
        else:
            print("Error: wrong model type")
            sys.exit()
    except KeyError as e:
        print(e)
        print("yaml file is probably wrong")
        sys.exit()
    model.load_weights(os.path.join(os.path.dirname(__file__), params['weights_path']))
    return model

def test_model(model, params, test_set, model_runfile_path, test_y, X):
    test_x = test_set[:TEST_PART_SIZE]

    recall, jaccard = evaluate_model_recall(model, test_x, test_y, params['topk'], X, fast_eval=False)

    result = "Model test performed \nRecall score: " + str(recall) + "     Jaccard score: " + str(jaccard)
    print(result)
    with open(model_runfile_path, 'a') as run_file:
        run_file.write("# " + result.replace("\n", "\n# "))

def main():
    try:
        model_runfile_path = sys.argv[1]
    except IndexError:
        print("Need to provide model run file.")
        sys.exit(1)

    if sys.argv[-1] == "vscode":
        vscode = True
    else:
        vscode = False

    if sys.argv[-2] != "notrain":
        train_model(model_runfile_path, vscode)

    params = read_params(model_runfile_path)

    test_set, test_y, X = get_test_set(params['dataset_name_prepend'])

    model = build_model(params, test_set.shape, X.shape[1])

    test_model(model, params, test_set, model_runfile_path, test_y, X)

if __name__ == "__main__":
	main()
