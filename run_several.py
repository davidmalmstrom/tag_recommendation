# Script to run several runs after each other. The runs are specified by the run-configurations 
# located in the folder runs/several_runs/

import sys
import os

sys.path.append('..')
import run_script

run_files = os.listdir("runs/several_runs/")
run_files.sort()

for run_file_name in run_files:
    try:
        run_script.main(["run_several.py", run_file_name], "runs/several_runs/")
    except Exception as e:
        print((str(type(e)) + ": " + str(e)).replace('\n', ' '))
#os.system("shutdown now -h")