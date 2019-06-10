# Script to run several runs after each other. The runs are specified by the run-configurations 
# located in the folder runs/several_runs/

import sys
import os
import traceback

sys.path.append('..')
import run_script

run_files = os.listdir("runs/several_runs/")

for run_file_name in run_files:
    try:
        run_script.main(["run_several.py", run_file_name], "runs/several_runs/")
    except Exception as e:
        traceback.print_exc()
        print(str(type(e)) + ": " + str(e))