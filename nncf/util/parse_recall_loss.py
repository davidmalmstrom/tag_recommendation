import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_line(line):
    iteration, rest = line.split('Iteration ')[1].split(' ', 1)
    recall, rest = rest.split('Recall = ')[1].split(',', 1)
    loss, _ = rest.split('loss = ')[1].split(',', 1)
    return [int(iteration), float(recall), float(loss)]

try:
    runfile_path = sys.argv[1]
    runfile_name = runfile_path.split('/')[-1]
except IndexError:
    print("Error: Runfile path required.")
    raise

with open(runfile_path) as f:
    data = [
        process_line(line)
        for line in f if '# Iteration ' in line
    ]

df = pd.DataFrame(data, columns=['Iteration', 'Recall@M', 'Loss'])

plt.figure()
zxc = sns.lineplot(x="Iteration", y="Recall@M",
             data=df)
plt.savefig('recall_graph_{}.svg'.format(runfile_name), bbox_inches='tight')

plt.figure()
zxc = sns.lineplot(x="Iteration", y="Loss",
             data=df)
plt.savefig('loss_graph_{}.svg'.format(runfile_name), bbox_inches='tight')
