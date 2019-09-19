import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_runfile(runfile_path):
    runfile_name = runfile_path.split('/')[-1]

    with open(runfile_path) as f:
        data = [
            process_line(line)
            for line in f if '# Iteration ' in line
        ]
        return data

def process_line(line):
    iteration, rest = line.split('Iteration ')[1].split(' ', 1)
    recall, rest = rest.split('Recall = ')[1].split(',', 1)
    loss, _ = rest.split('loss = ')[1].split(',', 1)
    return [int(iteration), float(recall), float(loss)]
