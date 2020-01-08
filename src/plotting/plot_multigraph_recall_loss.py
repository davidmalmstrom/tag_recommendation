import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJ_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(PROJ_ROOT_DIR)

from plotting.parse_recall_loss import parse_runfile

def get_df(model_name, runfile_path):
    data = parse_runfile(runfile_path)
    df = pd.DataFrame(data, columns = ['Epoch', 'Recall@10', 'Loss'])
    df['Model'] = model_name
    return df

run_files = {'GMF': os.path.join("deep_learning", "runs", "past_runs", "runz", "runz9.yml"),
             'MLP': os.path.join("deep_learning", "runs", "past_runs", "runza", "runza13.yml"),
             'Deep from scratch': os.path.join("deep_learning", "runs", "past_runs", "runzb", "runzb5.yml"),
             'Deep pretrained': os.path.join("deep_learning", "runs", "past_runs", "runza", "runza14.yml")
 }


frames = [
    get_df(model_name, os.path.join(PROJ_ROOT_DIR, runfile_path))
    for model_name, runfile_path in run_files.items()
]

min_length = min([len(frame) for frame in frames])
frames = [frame.head(min_length) for frame in frames]

df = pd.concat(frames)
plt.figure()
sns.set_palette('cubehelix', 4)

zxc = sns.lineplot(x="Epoch", y="Recall@10", hue='Model',
                data=df)

save_name = ''.join([
    path.split('/')[-1][:-4]
    for path in run_files.values()
])

fig_path = os.path.join(PROJ_ROOT_DIR, 'figures')

plt.savefig(os.path.join(fig_path, 'recall_graph_{}.svg'.format(save_name)), bbox_inches='tight')

plt.figure()
zxc = sns.lineplot(x="Epoch", y="Loss", hue='Model',
            data=df)
zxc.set(yscale="log")
plt.savefig(os.path.join(fig_path, 'loss_graph_{}.svg'.format(save_name)), bbox_inches='tight')
