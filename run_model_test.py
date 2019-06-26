import sys
import NeuMF
import MLP
import GMF
import oyaml as yaml
from evaluate_recall import evaluate_model_recall
import scipy.sparse as sp
import lib.notebook_helpers as nh
import pickle

try:
    model_runfile_path = sys.argv[1]
except IndexError:
    print("Need to provide model run file.")
    sys.exit(1)

with open(model_runfile_path) as run_file:
    yaml_params = yaml.safe_load(run_file)
    # yml file keys has leading "--"
    params_mid = {k[2:]: v for k, v in yaml_params.items()}
    params = {k: eval(v) if k in ['layers', 'reg_layers', 'num_factors', 'topk'] else v for k, v in params_mid.items()}

with open('Data/test_tag_dataset.pkl', 'rb') as f:
    _, y_test, _, _ = pickle.load(f)

# Just to get the dimensions for the model weight loading
with open('Data/dev_tag_dataset.pkl', 'rb') as f:
    _, y_train, _, _ = pickle.load(f)
    num_users, num_items = y_train.shape

# Build model
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

test_set = sp.dok_matrix(y_test)

val_x, val_y = nh.split_user_tags_percentage(test_set)

recall, jaccard = evaluate_model_recall(model, val_x, val_y, params['topk'])

result = "Model test performed \nRecall score: " + str(recall) + "     Jaccard score: " + str(jaccard)
print(result)
with open(model_runfile_path, 'a') as run_file:
    run_file.write("# " + result.replace("\n", "\n# "))