import os

# Change working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append('../')
from utils import *

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import ValidationCurveDisplay


import argparse
def main():
    """
    Command-line entry point for experiment automation.

    Parses arguments to select and run a stage function (e.g., hyperparams_search, validation_curve, evaluation).
    Use -e/--entry to specify the function to run. Additional key=value arguments can be passed if implemented.

    Example:
        python main.py -e hyperparams_search
    """
    parser = argparse.ArgumentParser(description="Run experiment stages.")
    parser.add_argument("-e", "--entry", required=True, help="Function to run")
    parser.add_argument("--kwargs", nargs='*', default=[], help="Additional key=value arguments")
    args = parser.parse_args()

    # Parse key=value pairs into a dict
    kwargs = {}
    for kv in args.kwargs:
        if '=' not in kv:
            raise ValueError(f"Invalid kwarg: {kv}. Use key=value format.")
        k, v = kv.split('=', 1)
        kwargs[k] = v

    if args.entry not in globals():
        raise ValueError(f"Unknown entry point: {args.entry}")
    globals()[args.entry](**kwargs)

if __name__ == "__main__":
    main()

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

    joblib.dump(grid, 'grid.joblib')

    marginals = plot_marginals(grid.cv_results_)

    for param, fig in marginals.items():
        fig.savefig(f'{param}_marginal.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

def validation_curve(
    param_range=np.logspace(0, 4, 20, base=10),
    n_splits=5,
    n_jobs=-1,
    verbose=10,
    est_params=dict(est__penalty='l2')
):
    (X_train, y_train), (X_test, y_test) = _load_data()
    grid = joblib.load('grid.joblib')
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

    plt.savefig('validation_curve.png', bbox_inches='tight', dpi=300)
    plt.close()

def evaluation(
    est_params=dict(est__penalty='l2', est__C=200)
):
    (X_train, y_train), (X_test, y_test) = _load_data()
    grid = joblib.load('grid.joblib')
    best_est = grid.best_estimator_

    best_est.set_params(**est_params)

    best_est.fit(X_train, y_train)
    joblib.dump(best_est, 'model.joblib')

    pr_fig, temporal_fig = plot_evals(best_est, X_test, y_test, y_train)
    pr_fig.savefig('precision_recall_curve.png', bbox_inches='tight', dpi=300)
    temporal_fig.savefig('temporal_eval.png', bbox_inches='tight', dpi=300)
    plt.close(pr_fig)
    plt.close(temporal_fig)
