import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append('./')
from utils import *
from gnn_model import GNNBinaryClassifier

import joblib
import json

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from torch_geometric.data import Data
from torch_geometric.nn import GCN

def _load_graph():
    nodes_df, edges_df = process_dataset()

    features = nodes_df.drop(columns=['time', 'class'])
    times = nodes_df['time']
    y = nodes_df['class']
    N_samples = len(y)

    data = Data(
        x=torch.tensor(features.values, dtype=torch.float),
        edge_index=torch.tensor(edges_df.values.T, dtype=torch.long),
        y=torch.tensor(y.values, dtype=torch.long),
        time=torch.tensor(times.values, dtype=torch.long)
    )

    train_val_idx, test_idx = temporal_split(data.time)

    labeled_mask = data.y != -1
    train_val_idx = train_val_idx[labeled_mask[train_val_idx]]
    test_idx = test_idx[labeled_mask[test_idx]]
    return data, (train_val_idx, test_idx)

def hyperparams_search(
    param_grid = {
        "est__hidden_channels": [16, 32, 64],
        "est__num_layers": [2, 3, 4],
        "est__dropout": [0.0, 0.2, 0.5],
        "est__lr": [0.001, 0.01, 0.1],
    },
    n_splits=5,
    n_jobs=-1,
    verbose=10,
):
    data, (train_val_idx, test_idx) = _load_graph()
    grid_cv = GridSearchCV(
        GNNBinaryClassifier(data, GCN),
        param_grid=param_grid,
        scoring='average_precision',
        cv=TemporalRollingCV(n_splits),
        n_jobs=n_jobs,
        verbose=verbose,
    )
    train_val_idx_np = train_val_idx.cpu().numpy()
    y = data.y.cpu().numpy()
    times = data.time.cpu().numpy()
    grid_cv.fit(train_val_idx_np, y[train_val_idx_np], groups=times[train_val_idx_np])

    joblib.dump(grid_cv, os.path.join(SCRIPT_DIR, 'gnn_grid_cv.joblib'))
    
    with open(os.path.join(SCRIPT_DIR, 'gnn_grid_cv_best_params.json'), 'w') as f:
        json.dump(grid_cv.best_params_, f, indent=2)    
    marginals = plot_marginals(grid_cv.cv_results_)
    for param, fig in marginals.items():
        fig.savefig(os.path.join(SCRIPT_DIR, f'{param}_marginal.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)
        
    
