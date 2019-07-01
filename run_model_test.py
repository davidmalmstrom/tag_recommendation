import sys
import NeuMF
import MLP
import GMF
import oyaml as yaml
from evaluate_recall import evaluate_model_recall
import scipy.sparse as sp
import lib.notebook_helpers as nh
import pickle
import run_script

TEST_PART_SIZE = 2000

def train_model(model_runfile_path):
    """Uses the NeuMF-module to build and train a model. This will modify the .yml file,
    adding the model weights path so that they can be loaded by the test_model functions.
    """
    read_params(model_runfile_path)  # for the assert
    run_script.main(["run_model_test.py", model_runfile_path])

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
        return params

def get_test_set():    
    with open('Data/test_tag_dataset.pkl', 'rb') as f:
        _, y_test, _, _, val_y = pickle.load(f)
        test_set = sp.dok_matrix(y_test)
    return test_set, val_y

def build_model(params, data_shape):
    num_users, num_items = data_shape
    try:
        if params['nn_model'] == 'NeuMF':
            reg_mf = 0
            model = NeuMF.get_model(num_users, num_items, params['num_factors'], params['layers'], params['reg_layers'], reg_mf)
        elif params['nn_model'] == "GMF":
            model = GMF.get_model(num_users, num_items, params['num_factors'])
        elif params['nn_model'] == "MLP":
            model = MLP.get_model(num_users, num_items, params['layers'], params['reg_layers'])
        else:
            print("Error: wrong model type")
            sys.exit()
    except KeyError as e:
        print(e)
        print("yaml file is probably wrong")
        sys.exit()
    model.load_weights(params['weights_path'])
    return model

def test_model(model, params, test_set, model_runfile_path, val_y):
    val_x = test_set[:TEST_PART_SIZE]

    recall, jaccard = evaluate_model_recall(model, val_x, val_y, params['topk'], fast_eval=False)

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

    train_model(model_runfile_path)

    params = read_params(model_runfile_path)

    test_set, val_y = get_test_set()

    model = build_model(params, test_set.shape)

    test_model(model, params, test_set, model_runfile_path, val_y)

if __name__ == "__main__":
	main()
