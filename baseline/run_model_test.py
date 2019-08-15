import sys
sys.path.append("..")

import base
import oyaml as yaml
import scipy.sparse as sp
import lib.notebook_helpers as nh
import pickle
from argparse import Namespace
from sklearn.metrics import jaccard_score, recall_score


TEST_PART_SIZE = 2000

def read_params(model_runfile_path):
    """Also asserts that the yml file is valid for testing
    """
    with open(model_runfile_path) as run_file:
        yaml_params = yaml.safe_load(run_file)

        params = base.parse_args(list(sum(list(yaml_params.items()), ())))  # transforms dict into list to be parsed

        try:
            assert(params.test_dataset == 1)
        except (KeyError, AssertionError) as e:
            print(repr(e))
            print("Not a valid test yml file; please specify \'--test_dataset: \"1\"\' in the yml-file.")
            sys.exit(1)
        try:
            assert(params.dataset_name_prepend)
        except (KeyError, AssertionError) as e:
            params.dataset_name_prepend = ""

        return params

def get_test_set(prepend):
    with open('../data/' + prepend + 'test_tag_dataset.pkl', 'rb') as f:
        X, y_test, _, _, _, test_y = pickle.load(f)
        test_set = y_test.tocsr()
    return test_set, test_y.toarray(), sp.csr_matrix(X)

def test_model(model, params, model_runfile_path, test_y, X):
    preds = model.predict(X[:TEST_PART_SIZE], start_index=0)

    recall = recall_score(test_y, preds, average='micro')
    jaccard = jaccard_score(test_y, preds, average='micro')

    result = ("\nModel test performed \nArguments: " + str(params) +
              "\nRecall score: " + str(recall) + "     Jaccard score: " + str(jaccard))
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

    params = read_params(model_runfile_path)

    test_set, test_y, X = get_test_set(params.dataset_name_prepend)

    model = base.get_model(params)

    model.fit(X, test_set)
    
    test_model(model, params, model_runfile_path, test_y, X)

if __name__ == "__main__":
	main()
