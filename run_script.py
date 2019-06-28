# Script to run the training, reads a YML-file with settings and writes output to it as comments
# Requires a json with the parameters in it

import os
import sys
import oyaml as yaml
import importlib

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

    nn_args = [item for sublist in params.items() for item in sublist]

    # Write stdout prints to end of yml file (commented out)
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(run_file_path + write_file_name, 'a')

        def write(self, m):
            if m == "\n":
                self.log.write(m)
            elif m[:2] == "!#":
                self.log.write(m[2:])
            else:
                self.log.write("# " + m)
            self.terminal.write(m)
            self.log.flush()

        def flush(self):
            self.log.flush()

    sys.stdout = Logger()
    print("\n")

    if args[-1] == "vscode":
        print("Launched by VS Code.")
    else:
        print("Launched by terminal.")

    import NeuMF
    NeuMF.main(nn_args)

    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main(sys.argv)