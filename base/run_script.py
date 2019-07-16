# Script to run the training, reads a YML-file with settings and writes output to it as comments
# Requires a json with the parameters in it

import sys
sys.path.append("..")

import os
import oyaml as yaml
import importlib
import lib.utils as utils

def main(args, run_file_path=None):
    if run_file_path is None:
        run_file_path = ""

    if len(args) < 2 or ".yml" not in args[1]:
        print("No YML parameter configuration file provided, using default run_template.")
        read_file_name = "run_template.yml"
        i = 1
        while os.path.exists(run_file_path + "run%s.yml" % i):
            i += 1
        write_file_name = "run%s.yml" % i
    else:
        read_file_name = args[1]
        write_file_name = read_file_name

    if read_file_name == "run_template.yml":
        path = "runs/" + read_file_name
    else:
        path = run_file_path + read_file_name

    with open(path) as yaml_file:
        params = yaml.safe_load(yaml_file)

    if read_file_name == "run_template.yml":
        with open(run_file_path + write_file_name, 'w') as w_file:
            yaml.dump(params, w_file)

    model_args = [item for sublist in params.items() for item in sublist]

    sys.stdout = utils.Logger(run_file_path + write_file_name)
    print("\n")

    if args[-1] == "vscode":
        print("Launched by VS Code.")
    else:
        print("Launched by terminal.")

    import base
    base.main(model_args)

    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main(sys.argv)