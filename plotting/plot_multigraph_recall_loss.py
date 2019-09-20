import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJ_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(PROJ_ROOT_DIR)

from plotting.parse_recall_loss import parse_runfile

def get_df(model_name, runfile_path):
    data = parse_runfile(runfile_path)
    df = pd.DataFrame(data, columns = ['Iteration', 'Recall@M', 'Loss'])
    df['Model'] = model_name
    return df

run_files = {'GMF': os.path.join("nncf", "runs", "past_runs", "runy", "runy16.yml"),
             'MLP': os.path.join("nncf", "runs", "past_runs", "runy", "runy17.yml"),
             'NN from scratch': os.path.join("nncf", "runs", "past_runs", "runy", "runy18.yml"),
             'NN pretrained': os.path.join("nncf", "runs", "past_runs", "runy", "runy8.yml")
 }


frames = [
    get_df(model_name, os.path.join(PROJ_ROOT_DIR, runfile_path))
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

fig_path = os.path.join(PROJ_ROOT_DIR, 'figures')

plt.savefig(os.path.join(fig_path, 'recall_graph_{}.svg'.format(save_name)), bbox_inches='tight')

plt.figure()
zxc = sns.lineplot(x="Iteration", y="Loss", hue='Model',
            data=df)
zxc.set(yscale="log")
plt.savefig(os.path.join(fig_path, 'loss_graph_{}.svg'.format(save_name)), bbox_inches='tight')
