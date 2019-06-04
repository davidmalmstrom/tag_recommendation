# Script to run the training, reads a json with settings and writes output to it as comments
# Requires a json with the parameters in it

import os
import sys
import oyaml as yaml
import importlib

if len(sys.argv) < 2:
    print("No JSON parameter configuration file provided, using default run_template.")
    read_file_name = "run_template.yml"
    i = 1
    while os.path.exists("runs/run%s.yml" % i):
        i += 1
    write_file_name = "run%s.yml" % i
else:
    read_file_name = sys.argv[1]
    write_file_name = read_file_name

with open("runs/" + read_file_name) as yaml_file:
    params = yaml.safe_load(yaml_file)

if read_file_name == "run_template.yml":
    with open("runs/" + write_file_name, 'w') as w_file:
        yaml.dump(params, w_file)

args = [item for sublist in params.items() for item in sublist]

# Write stdout prints to end of json file (commented out)
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("runs/" + write_file_name, 'a')

    def write(self, m):
        if m == "\n":
            self.log.write(m)
        else:
            self.log.write("# " + m)
        self.terminal.write(m)
        self.log.flush()

    def flush(self):
        self.log.flush()

sys.stdout = Logger()
print("\n")

import NeuMF
NeuMF.main(args)
