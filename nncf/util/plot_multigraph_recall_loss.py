import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")
from nncf.util.parse_recall_loss import parse_runfile

run_files = {'GMF': 'runs/past_runs/runw/runw1.yml',
             'MLP': 'runs/past_runs/runx/runx5.yml',
             'NN from scratch': 'runs/past_runs/runx/runx99_neumfdummy.yml',
             'NN pretrained': 'runs/past_runs/runy/runy8.yml'
 }


def get_df(model_name, runfile_path):
    data = parse_runfile(runfile_path)
    df = pd.DataFrame(data, columns = ['Iteration', 'Recall@M', 'Loss'])
    df['Model'] = model_name
    return df

frames = [
    get_df(model_name, runfile_path)
    for model_name, runfile_path in run_files.items()
]

df = pd.concat(frames)
plt.figure()
zxc = sns.lineplot(x="Iteration", y="Recall@M", hue='Model',
                data=df)

save_name = ''.join([
    path.split('/')[-1][:-4]
    for path in run_files.values()
])

plt.savefig('recall_graph_{}.svg'.format(save_name), bbox_inches='tight')

plt.figure()
zxc = sns.lineplot(x="Iteration", y="Loss", hue='Model',
            data=df)
zxc.set(yscale="log")
plt.savefig('loss_graph_{}.svg'.format(save_name), bbox_inches='tight')
