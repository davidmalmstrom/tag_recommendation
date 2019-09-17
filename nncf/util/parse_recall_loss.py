import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_line(line):
    iteration, rest = line.split('Iteration ')[1].split(' ', 1)
    recall, rest = rest.split('Recall = ')[1].split(',', 1)
    loss, _ = rest.split('loss = ')[1].split(',', 1)
    return [int(iteration), float(recall), float(loss)]

with open(sys.argv[1]) as f:
    data = [
        process_line(line)
        for line in f if '# Iteration ' in line
    ]

df = pd.DataFrame(data, columns=['iteration', 'recall', 'loss'])

plt.figure()
zxc = sns.lineplot(x="iteration", y="recall",
             data=df)
plt.savefig('recall_graph.svg', bbox_inches='tight')

plt.figure()
zxc = sns.lineplot(x="iteration", y="loss",
             data=df)
plt.savefig('loss_graph.svg', bbox_inches='tight')
