# Script to run several runs after each other. The runs are specified by the run-configurations
# located in the folder runs/several_runs/

import sys
import os

sys.path.append('..')
import run_script

files_path = "runs/several_runs/"

run_files = os.listdir(files_path)
run_files.remove('.keep')
run_files.sort()
seen = set()

while seen != set(run_files):
    for run_file_name in run_files:
        if run_file_name not in seen:
            try:
                seen.add(run_file_name)
                run_script.main(["run_several.py", run_file_name], "runs/several_runs/")
            except Exception as e:
                print((str(type(e)) + ": " + str(e)).replace('\n', ' '))
                sys.stdout = sys.stdout.terminal  # Reset stdout so that new prints
                                                    # does not go into old file
            break

    run_files = os.listdir(files_path)
    run_files.sort()

# os.system("shutdown now -h")  # run with sudo nohup /home/david/miniconda2/envs/t10/bin/python run_several.py
# and with ctrl+Z and bg %1
