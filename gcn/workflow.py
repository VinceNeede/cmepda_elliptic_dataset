import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append('./')
from utils import *
from gnn_model import GNNBinaryClassifier

import joblib
import json
import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
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
    param_distributions = {
        "hidden_dim": [16, 32, 64],
        "num_layers": randint(2, 6),
        "dropout": uniform(0.0, 0.8),
        "learning_rate_init": loguniform(1e-4, 1e-1),
        "weight_decay": loguniform(1e-6, 1e-2),
    },
    n_iter=64,
    n_splits=5,
    n_jobs=1,  # Use single job to avoid memory issues with large graphs
    verbose=10,
    max_iter=1000,
):
    data, (train_val_idx, test_idx) = _load_graph()
    rand_cv = RandomizedSearchCV(
        GNNBinaryClassifier(data, GCN, max_iter=max_iter),
        param_distributions=param_distributions,
        scoring='average_precision',
        cv=TemporalRollingCV(n_splits),
        n_jobs=n_jobs,
        verbose=verbose,
        n_iter=n_iter,
    )
    train_val_idx_np = train_val_idx.cpu().numpy()
    y = data.y.cpu().numpy()
    times = data.time.cpu().numpy()
    rand_cv.fit(train_val_idx_np, y[train_val_idx_np], groups=times[train_val_idx_np])

    joblib.dump(rand_cv, os.path.join(SCRIPT_DIR, 'gnn_rand_cv.joblib'))

    with open(os.path.join(SCRIPT_DIR, 'gnn_rand_cv_best_params.json'), 'w') as f:
        json.dump(rand_cv.best_params_, f, indent=2)
    marginals = plot_marginals(rand_cv.cv_results_)
    for param, fig in marginals.items():
        fig.savefig(os.path.join(SCRIPT_DIR, f'{param}_marginal.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)
        
    
