import os
from tkinter import Y
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append('./')
from utils import *

import joblib
import json

import matplotlib.pyplot as plt


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

def hyperparams_search(
    distributions = {
        "est__n_estimators": randint(10, 200),
        "est__max_depth": [None, 10, 20, 30],
        "est__min_samples_split": randint(2, 10),
        "est__class_weight": [None, "balanced", "balanced_subsample"],
        "est__max_features": uniform(0.1, 0.9)
    },
    n_iter=50,
    n_splits=5,
    n_jobs=-1,
    verbose=10
):
    (X_train, y_train), (X_test, y_test) = load_labeled_data()
    rand_cv = RandomizedSearchCV(
        Pipeline([
            ('drop_time', DropTime()),
            ('est', RandomForestClassifier()),
        ]),
        param_distributions=distributions,
        n_iter=n_iter,
        scoring=pr_auc_scorer,
        cv=TemporalRollingCV(n_splits=n_splits),
        n_jobs=n_jobs,
        verbose=verbose,
    )
    rand_cv.fit(X_train, y_train)
    joblib.dump(rand_cv, os.path.join(SCRIPT_DIR, 'rf_rand_cv.joblib'))
    with open(os.path.join(SCRIPT_DIR, 'rf_rand_cv_best_params.json'), 'w') as f:
        json.dump(rand_cv.best_params_, f, indent=2)
        
    marginals = plot_marginals(rand_cv.cv_results_)
    for param, fig in marginals.items():
        fig.savefig(os.path.join(SCRIPT_DIR, f'{param}_marginal.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

def evaluation():
    (X_train, y_train), (X_test, y_test) = load_labeled_data()
    rand_cv = joblib.load(os.path.join(SCRIPT_DIR, 'rf_rand_cv.joblib'))
    best_est = rand_cv.best_estimator_

    pr_fig, temporal_fig = plot_evals(best_est, X_test, y_test, y_train)
    pr_fig.savefig(os.path.join(SCRIPT_DIR, 'precision_recall_curve.png'), bbox_inches='tight', dpi=300)
    temporal_fig.savefig(os.path.join(SCRIPT_DIR, 'temporal_eval.png'), bbox_inches='tight', dpi=300)
    plt.close(pr_fig)
    plt.close(temporal_fig)