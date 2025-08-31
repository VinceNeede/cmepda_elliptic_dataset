import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append('./')
from utils import *

import joblib
import json

import matplotlib.pyplot as plt


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ValidationCurveDisplay

def _load_data():
    nodes_df, edges_df = process_dataset()
    nodes_df = nodes_df[nodes_df['class'] != -1] # select only labeled data
    (X_train, y_train), (X_test, y_test) = temporal_split(nodes_df, test_size=0.2)
    return (X_train, y_train), (X_test, y_test)

def hyperparams_search(
    param_grid={
        "est__C": np.logspace(-4, 4, 10),
        "est__penalty": ["l1", "l2"],
        "est__class_weight": [None, "balanced"]
    },
    max_iter=5000,
    n_splits=5,
    n_jobs=-1,
    verbose=10
):
    """ =========== LOAD DATA =========== """
    (X_train, y_train), (X_test, y_test) = _load_data()

    grid = GridSearchCV(
        Pipeline([
            ('drop_time', DropTime()),
            ('est', LogisticRegression(max_iter=max_iter, solver='liblinear')),        
        ]),
        param_grid=param_grid,
        scoring=pr_auc_scorer,
        cv=TemporalRollingCV(n_splits=n_splits),
        n_jobs=n_jobs,
        verbose=verbose
    )

    grid.fit(X_train, y_train)

    joblib.dump(grid, os.path.join(SCRIPT_DIR, 'grid.joblib'))

    with open(os.path.join(SCRIPT_DIR, 'grid_best_params.json'), 'w') as f:
        json.dump(grid.best_params_, f, indent=2)

    marginals = plot_marginals(grid.cv_results_)

    for param, fig in marginals.items():
        fig.savefig(os.path.join(SCRIPT_DIR, f'{param}_marginal.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

def validation_curve(
    param_range=np.logspace(0, 4, 20, base=10),
    n_splits=5,
    n_jobs=-1,
    verbose=10,
    est_params=dict(est__penalty='l2')
):
    (X_train, y_train), (X_test, y_test) = _load_data()
    grid = joblib.load(os.path.join(SCRIPT_DIR, 'grid.joblib'))
    best_est = grid.best_estimator_

    best_est.set_params(**est_params)

    ValidationCurveDisplay.from_estimator(
        best_est,
        X_train,
        y_train,
        param_name="est__C",
        param_range=param_range,
        n_jobs=n_jobs,
        scoring=pr_auc_scorer,
        score_name="PR AUC",
        cv=TemporalRollingCV(n_splits=n_splits),
        verbose=verbose,
    )

    plt.savefig(os.path.join(SCRIPT_DIR, 'validation_curve.png'), bbox_inches='tight', dpi=300)
    plt.close()

def evaluation(
    est_params=dict(est__penalty='l2', est__C=200)
):
    (X_train, y_train), (X_test, y_test) = _load_data()
    grid = joblib.load(os.path.join(SCRIPT_DIR, 'grid.joblib'))
    best_est = grid.best_estimator_

    best_est.set_params(**est_params)

    best_est.fit(X_train, y_train)
    joblib.dump(best_est, os.path.join(SCRIPT_DIR, 'model.joblib'))

    pr_fig, temporal_fig = plot_evals(best_est, X_test, y_test, y_train)
    pr_fig.savefig(os.path.join(SCRIPT_DIR, 'precision_recall_curve.png'), bbox_inches='tight', dpi=300)
    temporal_fig.savefig(os.path.join(SCRIPT_DIR, 'temporal_eval.png'), bbox_inches='tight', dpi=300)
    plt.close(pr_fig)
    plt.close(temporal_fig)

