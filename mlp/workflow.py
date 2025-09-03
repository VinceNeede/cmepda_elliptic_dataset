import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append('./')
from utils import *
from sklearn.metrics import average_precision_score, make_scorer

import joblib
import json

import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ValidationCurveDisplay
from scipy.stats import uniform, loguniform, randint

def _load_data():
    nodes_df, edges_df = process_dataset()
    nodes_df = nodes_df[nodes_df['class'] != -1] # select only labeled data
    (X_train, y_train), (X_test, y_test) = temporal_split(nodes_df, test_size=0.2)
    return (X_train, y_train), (X_test, y_test)

class MLPWrapper(MLPClassifier):
    def __init__(
        self,
        num_layers=2,
        hidden_dim=16,
        alpha=0.0001,
        learning_rate_init=0.001,
        batch_size='auto',
        max_iter=1000,
        ):
        hidden_layer_sizes = tuple([hidden_dim]*num_layers)
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            batch_size=batch_size,
            shuffle=False,
            early_stopping=False,
        )
        

def hyperparams_search(
    param_distributions={
        "est__num_layers": randint(1, 5),
        "est__hidden_dim": [16, 32, 64, 128],
        "est__alpha": loguniform(1e-6, 1e-1),
        "est__learning_rate_init": loguniform(1e-4, 1e-1),
    },
    max_iter=1000,
    n_iter=50,
    n_splits=5,
    n_jobs=-1,
    verbose=10
):
    """ =========== LOAD DATA =========== """
    (X_train, y_train), (X_test, y_test) = _load_data()

    rand_cv = RandomizedSearchCV(
        Pipeline([
            ('drop_time', DropTime()),
            ('est', MLPWrapper(max_iter=max_iter)),
        ]),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=make_scorer(average_precision_score, response_method="predict_proba"),
        cv=TemporalRollingCV(n_splits=n_splits),
        n_jobs=n_jobs,
        verbose=verbose,
    )
    
    rand_cv.fit(X_train, y_train)
    
    joblib.dump(rand_cv, os.path.join(SCRIPT_DIR, 'mlp_rand_cv.joblib'))
    
    with open(os.path.join(SCRIPT_DIR, 'mlp_rand_cv_best_params.json'), 'w') as f:
        json.dump(rand_cv.best_params_, f, indent=2)
        
    marginals = plot_marginals(rand_cv.cv_results_)
    for param, fig in marginals.items():
        fig.savefig(os.path.join(SCRIPT_DIR, f'{param}_marginal.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

def evaluation():
    """ =========== LOAD DATA =========== """
    (X_train, y_train), (X_test, y_test) = _load_data()
    rand_cv = joblib.load(os.path.join(SCRIPT_DIR, 'mlp_rand_cv.joblib'))
    best_est = rand_cv.best_estimator_

    best_est.fit(X_train, y_train)
    joblib.dump(best_est, os.path.join(SCRIPT_DIR, 'mlp_model.joblib'))

    pr_fig, temporal_fig = plot_evals(best_est, X_test, y_test, y_train)
    pr_fig.savefig(os.path.join(SCRIPT_DIR, 'precision_recall_curve.png'), bbox_inches='tight', dpi=300)
    temporal_fig.savefig(os.path.join(SCRIPT_DIR, 'temporal_eval.png'), bbox_inches='tight', dpi=300)
    plt.close(pr_fig)
    plt.close(temporal_fig)
