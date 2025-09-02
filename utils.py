
# Standard library imports
import os
import warnings
from functools import singledispatch, partial

# Third-party imports
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, auc, make_scorer
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# Project-specific imports (none in this file)

def process_dataset(
    folder_path: str = "elliptic_bitcoin_dataset",
    features_file: str = "elliptic_txs_features.csv",
    classes_file: str = "elliptic_txs_classes.csv",
    edges_file: str = "elliptic_txs_edgelist.csv",
):
    """
    Loads, validates, and processes the Elliptic Bitcoin dataset.

    Returns
    -------
    nodes_df : pandas.DataFrame
        DataFrame with shape (203769, 167). Columns:
            - 'time': Discrete time step (int)
            - 'feat_0' ... 'feat_164': Node features (float)
            - 'class': Node label (int: 1 for illicit, 0 for licit, -1 for unknown/missing)
        The 'class' column uses -1 to indicate missing labels (transductive setting).
        The 'txId' column is dropped in the returned DataFrame; its original order matches the input file.

    edges_df : pandas.DataFrame
        DataFrame with shape (234355, 2). Columns:
            - 'txId1': Source node index (int, row index in nodes_df)
            - 'txId2': Target node index (int, row index in nodes_df)
        Each row represents a directed edge in the transaction graph, with node indices corresponding to rows in nodes_df.

    Notes
    -----
    - All IDs in 'edges_df' are mapped to row indices in 'nodes_df'.
    - The function performs strict validation on shapes, unique values, and label distribution.
    - The returned DataFrames are ready for use in graph ML frameworks (e.g., PyTorch Geometric, DGL).
    """
    classes_path = os.path.join(folder_path, classes_file)
    features_path = os.path.join(folder_path, features_file)
    edges_path = os.path.join(folder_path, edges_file)

    classes_df = pd.read_csv(classes_path)
    features_df = pd.read_csv(features_path, header=None)
    edges_df = pd.read_csv(edges_path)
    # Basic checks
    
    # features checks
    assert features_df.shape == (203769, 167)
    assert features_df[0].nunique() == 203769  # txId is unique
    assert features_df[1].nunique() == 49  # time has 49 unique values
    
    # classes checks
    assert all(classes_df.columns == ['txId', 'class'])
    assert classes_df.shape == (203769, 2)
    assert set(classes_df['class'].unique()) == set(['unknown', '1', '2'])
    classes_counts = classes_df['class'].value_counts()
    assert classes_counts['unknown'] == 157205
    assert classes_counts['1'] == 4545
    assert classes_counts['2'] == 42019
    assert set(classes_df['txId']) == set(features_df[0])

    # edges checks
    assert edges_df.shape == (234355, 2)
    assert all(edges_df.columns == ['txId1', 'txId2'])
    assert set(edges_df['txId1']).issubset(set(features_df[0]))
    assert set(edges_df['txId2']).issubset(set(features_df[0]))
    
    features_names = ['txId', 'time'] + [f'feat_{i}' for i in range(165)]
    features_df.columns = features_names
    
    class_map = {'unknown': -1, '1': 1, '2': 0}
    classes_df['class'] = classes_df['class'].map(class_map)
    
    nodes_df = features_df.join(classes_df.set_index('txId')['class'], on='txId', how='left')
    
    txid_to_idx = pd.Series(nodes_df.index, index=nodes_df['txId'])

    # Map txId1 and txId2 in edges_df to node indices
    edges_df['txId1'] = edges_df['txId1'].map(txid_to_idx)
    edges_df['txId2'] = edges_df['txId2'].map(txid_to_idx)
    
    return nodes_df.drop(columns=['txId']), edges_df

class TemporalRollingCV(TimeSeriesSplit):
    """
    Time-based cross-validation iterator that extends scikit-learn's TimeSeriesSplit
    to work with data that has explicit time step values (like the Elliptic Bitcoin dataset).
    
    This class inherits from TimeSeriesSplit and adds functionality to handle datasets
    where multiple samples can belong to the same time step. It maps the time step indices
    to actual row indices in the dataset, allowing it to be used with datasets like
    the Elliptic Bitcoin dataset.
    
    This CV strategy ensures that for each fold:
    1. Training data comes from earlier time periods
    2. The test set is a continuous time window following the training data
    3. Each fold expands the training window and shifts the test window forward
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of splits to generate
    test_size : int, default=None
        Size of test window in time steps. If None, will be calculated based on n_splits.
    max_train_size : int, default=None
        Maximum number of time steps to use for training. If None, all available time steps
        will be used.
    gap : int, default=0
        Number of time steps to skip between training and test sets
    time_col : str, default='time'
        Name of the column containing time step information
        
    Examples:
    ---------
    >>> # Basic usage
    >>> from elliptic_utils.data_loading import load_dataset
    >>> df, _ = load_dataset("./elliptic_bitcoin_dataset")
    >>> cv = TemporalRollingCV(n_splits=3, test_size=1)
    >>> for train_idx, test_idx in cv.split(df):
    ...     print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
    ...     # Note: train_idx and test_idx are indices of rows, not time steps
    ...     train_times = sorted(df.iloc[train_idx]['time'].unique())
    ...     test_times = sorted(df.iloc[test_idx]['time'].unique())
    ...     print(f"  Train time steps: {train_times}")
    ...     print(f"  Test time steps: {test_times}")
    >>> 
    >>> # With GridSearchCV
    >>> from sklearn.model_selection import GridSearchCV
    >>> from elliptic_utils.evaluation import pr_auc_scorer
    >>> cv = TemporalRollingCV(n_splits=3, test_size=1)
    >>> grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=pr_auc_scorer)
    >>> grid_search.fit(df, df['class'])
    """
    def __init__(self, n_splits=5, *, test_size=None, max_train_size=None, gap=0, time_col='time'):
        super().__init__(n_splits=n_splits, test_size=test_size, max_train_size=max_train_size, gap=gap)
        self.time_col = time_col

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.
        
        Unlike standard TimeSeriesSplit, this method works with explicit time step values
        and maps them to actual row indices in the dataset. This allows it to handle
        datasets where multiple samples can belong to the same time step.
        
        Parameters:
        -----------
        X : array-like, DataFrame
            Training data. If DataFrame, must contain the column specified by `time_col`.
            Otherwise, time values must be passed through the `groups` parameter.
        y : array-like, optional
            Targets for the training data (ignored)
        groups : array-like, optional
            Time values for each sample if X doesn't have the time column specified by time_col
            
        Yields:
        -------
        train_index : ndarray
            Indices of rows in the training set
        test_index : ndarray
            Indices of rows in the test set
            
        Notes:
        ------
        The yielded indices refer to rows in the original dataset, not time steps.
        This makes the cross-validator compatible with scikit-learn's model selection tools.
        """
        # Get time values
        if hasattr(X, self.time_col) and isinstance(getattr(X, self.time_col), pd.Series):
            times = getattr(X, self.time_col).values
        elif groups is not None:
            times = groups
        else:
            raise ValueError(f"X must have a '{self.time_col}' column or time values must be passed as groups")
        
        if isinstance(times, np.ndarray) or isinstance(times, pd.Series):
            mod = np
        elif isinstance(times, torch.Tensor):
            mod = torch
        else:
            raise ValueError("times must be a numpy array, torch tensor, or pandas Series")        

        # Get unique time steps and sort
        unique_times = mod.unique(times)
        for train_times, test_times in super().split(unique_times):
            train_mask = mod.isin(times, unique_times[train_times])
            test_mask = mod.isin(times, unique_times[test_times])
            train_indices = mod.where(train_mask)[0]
            test_indices = mod.where(test_mask)[0]
            yield train_indices, test_indices

@singledispatch
def temporal_split(times, test_size=0.2):
    """
    Split data into temporal train/test sets based on unique time steps.

    Parameters
    ----------
    times : np.ndarray, torch.Tensor, or pandas.DataFrame
        The time information or data to split. For DataFrames, must contain a 'time' column.
    test_size : float, default=0.2
        Proportion of unique time steps to include in the test split (between 0.0 and 1.0).

    Returns
    -------
    For array/tensor input:
        train_indices, test_indices : array-like
            Indices for training and test sets.
    For DataFrame input:
        (X_train, y_train), (X_test, y_test) : tuple of tuples
            X_train : pandas.DataFrame
                Training features (all columns except 'class').
            y_train : pandas.Series
                Training labels (the 'class' column).
            X_test : pandas.DataFrame
                Test features (all columns except 'class').
            y_test : pandas.Series
                Test labels (the 'class' column).
        Or, if return_X_y=False:
            train_df, test_df : pandas.DataFrame
                The full training and test DataFrames, already sliced by time.

    Type-specific behavior
    ---------------------
    - np.ndarray: Uses numpy operations to split by unique time values.
    - torch.Tensor: Uses torch operations to split by unique time values (no CPU/GPU transfer).
    - pandas.DataFrame: Splits based on the 'time' column. If return_X_y=True, unpacks X and y based on the 'class' column; otherwise, returns the sliced DataFrames.

    """
    raise NotImplementedError("temporal_split not implemented for this type")

def _temporal_split(times, mod, test_size):
    """
    Core logic for temporal splitting, used by temporal_split for both numpy and torch arrays.
    Issues a warning if n_train or n_test is zero.
    Parameters
    ----------
    times : array-like
        Array of time values (numpy or torch).
    mod : module
        Module to use (np or torch) for unique, isin, where.
    test_size : float
        Proportion of unique time steps to include in the test split.
    Returns
    -------
    train_indices, test_indices : array-like
        Indices for training and test sets.
    """
    unique_times = mod.unique(times)
    n_test = int(len(unique_times) * test_size)
    n_train = len(unique_times) - n_test

    if n_train == 0 or n_test == 0:
        msg = (
            f"temporal_split: n_train or n_test is zero. "
            f"n_train={n_train}, n_test={n_test}, total unique_times={len(unique_times)}. "
            f"Check your test_size ({test_size}) and data."
        )
        if n_train == 0:
            msg += " All data assigned to test set."
        if n_test == 0:
            msg += " All data assigned to train set."
        warnings.warn(msg)

    train_times = unique_times[:n_train]
    test_times = unique_times[n_train:]
    train_mask = mod.isin(times, train_times)
    test_mask = mod.isin(times, test_times)

    train_indices = mod.where(train_mask)[0]
    test_indices = mod.where(test_mask)[0]
    return train_indices, test_indices

@temporal_split.register(np.ndarray)
def _(times, test_size=0.2):
    """
    Temporal split for numpy arrays.
    See _temporal_split for details.
    """
    return _temporal_split(times, np, test_size)

@temporal_split.register(torch.Tensor)
def _(times, test_size=0.2):
    """
    Temporal split for torch tensors.
    See _temporal_split for details.
    """
    return _temporal_split(times, torch, test_size)

@temporal_split.register(pd.DataFrame)
def _(nodes_df, test_size=0.2, return_X_y=True):
    """
    Temporal split for pandas DataFrames.
    Splits based on the 'time' column. If return_X_y=True, returns (X_train, y_train), (X_test, y_test) tuples;
    otherwise, returns the full train/test DataFrames.
    """
    train_indices, test_indices = temporal_split(nodes_df['time'].values, test_size=test_size)

    train_df = nodes_df.iloc[train_indices].reset_index(drop=True)
    test_df = nodes_df.iloc[test_indices].reset_index(drop=True)

    if not return_X_y:
        return train_df, test_df
    X_train, y_train = train_df.drop(columns=['class']), train_df['class']
    X_test, y_test = test_df.drop(columns=['class']), test_df['class']
    return (X_train, y_train), (X_test, y_test)

def load_labeled_data():
    """
    Loads the processed dataset and returns only labeled data, split temporally into train and test sets.
    Returns
    -------
    (X_train, y_train), (X_test, y_test) : tuple of tuples
        X_train, y_train: training features and labels
        X_test, y_test: test features and labels
    """
    nodes_df, edges_df = process_dataset()
    nodes_df = nodes_df[nodes_df['class'] != -1] # select only labeled data
    (X_train, y_train), (X_test, y_test) = temporal_split(nodes_df, test_size=0.2)
    return (X_train, y_train), (X_test, y_test)

def pr_auc_score(y_true, y_pred_proba):
    """
    Calculate Precision-Recall AUC score.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for the positive class
        
    Returns:
    --------
    float
        PR AUC score
        
    Examples:
    ---------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X, y = make_classification(random_state=42)
    >>> clf = RandomForestClassifier(random_state=42).fit(X, y)
    >>> y_prob = clf.predict_proba(X)[:, 1]
    >>> pr_auc_score(y, y_prob)
    """
    # Handle case where y_pred_proba is a 2D array (output from predict_proba)
    if hasattr(y_pred_proba, 'ndim') and y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]
    
    # Calculate precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    return pr_auc

# Create a scorer for use with GridSearchCV
pr_auc_scorer = make_scorer(pr_auc_score, greater_is_better=True, response_method="predict_proba")

import matplotlib.pyplot as plt

def _get_marginals_ticks(x_labels, N=10):
    """
    Helper for plot_marginals: reduces the number of x-ticks for readability.
    """
    x = list(range(len(x_labels)))  # or use the actual tick positions if available
    if len(x_labels) > N:
        step = max(1, len(x_labels) // N)
        shown_idx = list(range(0, len(x_labels), step))
        
        return x[shown_idx], x_labels[shown_idx]
    return x, x_labels

def plot_marginals(cv_results, max_ticks=10):
    """
    Plot marginal mean test scores for each hyperparameter in cv_results.

    Parameters
    ----------
    cv_results : dict or DataFrame
        The cv_results_ attribute from a scikit-learn search object.

    max_ticks : int, default=10
        Maximum number of x-ticks to show on the x-axis for readability.

    Returns
    -------
    figs : dict
        Dictionary mapping parameter names to matplotlib.figure.Figure objects.
    """
    results = pd.DataFrame(cv_results)
    param_names = [col for col in results.columns if col.startswith('param_')]
    figs = dict()
    for param in param_names:
        fig, ax = plt.subplots()
        # Group by the parameter and compute mean test score
        marginals = results.groupby(param, dropna=False)['mean_test_score'].mean()
        marginals_std = results.groupby(param, dropna=False)['std_test_score'].mean()
        x_labels = [f"{x:.2g}" if isinstance(x, float) else str(x) for x in marginals.index]
        ax.errorbar(x_labels, marginals, yerr=marginals_std, fmt='-o')
        ax.set_title(f'Marginal mean test score for {param}')
        ax.set_ylabel('Mean test score')
        xticks, xticks_labels = _get_marginals_ticks(x_labels, max_ticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labels, rotation=45)
        figs[param] = fig
    return figs

from sklearn.metrics import PrecisionRecallDisplay

def plot_evals(est, X_test, y_test, y_train,*, time_steps_test=None):
    """
    Generate two evaluation plots for a classifier:
    1. Precision-Recall curve on the test set.
    2. Rolling/cumulative PR AUC and illicit rate by time step.

    Parameters
    ----------
    est : classifier
        Trained classifier with predict_proba and predict methods.
    X_test : pd.DataFrame
        Test features. Must contain a 'time' column unless time_steps_test is provided.
    y_test : array-like
        Test labels (binary).
    y_train : array-like
        Training labels (binary), used for reference illicit rate.
    time_steps_test : array-like, optional
        Time step values for test set. If None, will use X_test['time'].

    Returns
    -------
    pr_fig : matplotlib.figure.Figure
        Figure for the precision-recall curve.
    temporal_fig : matplotlib.figure.Figure
        Figure for the rolling/cumulative PR AUC and illicit rate by time step.
    """
    y_pred_proba = est.predict_proba(X_test)[:, 1]
    y_pred = est.predict(X_test)

    pr_fig, pr_ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, plot_chance_level=True, ax=pr_ax)

    # Get time steps for test data
    if time_steps_test is None:
        if hasattr(X_test, 'time'):
            time_steps_test = X_test['time'].values
        else:
            raise ValueError('either pass time_steps_test esplicitly or X_test must have column `time`')

    # Create results DataFrame
    results_df = pd.DataFrame({
        'time': time_steps_test,
        'actual': y_test,
        'pred_proba': y_pred_proba,
    })

    # Get unique time steps in ascending order
    unique_times = sorted(results_df['time'].unique())

    # Prepare data structures for rolling analysis
    rolling_metrics = []

    # For each cutoff point, calculate metrics on all data up to and including that time step
    for i, cutoff_time in enumerate(unique_times):
        # Select data up to and including the current time step
        current_data = results_df[results_df['time'] <= cutoff_time]
        
        current_pr_auc = pr_auc_score(current_data['actual'], current_data['pred_proba'])
        current_illicit_rate = np.mean(current_data['actual'] == 1)
        
        rolling_metrics.append({
            'cutoff_time': cutoff_time,
            'pr_auc': current_pr_auc,
            'illicit_rate': current_illicit_rate,
            'sample_size': len(current_data),
            'illicit_count': sum(current_data['actual'] == 1)
        })

    # Convert to DataFrame
    rolling_df = pd.DataFrame(rolling_metrics)

    # Calculate training set illicit rate for reference
    train_illicit_rate = np.mean(y_train == 1)

    # Create figure with two y-axes
    temporal_fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot PR AUC on primary y-axis
    ax1.plot(rolling_df['cutoff_time'], rolling_df['pr_auc'], 'b-o', linewidth=2, 
                        label='Rolling PR AUC')
    ax1.set_xlabel('Time Step Cutoff', fontsize=12)
    ax1.set_ylabel('PR AUC', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)

    # Create a secondary y-axis for illicit rates
    ax2 = ax1.twinx()
    ax2.plot(rolling_df['cutoff_time'], rolling_df['illicit_rate'], 'r-^', linewidth=2,
                    label='Rolling Illicit Rate')
    ax2.axhline(y=train_illicit_rate, color='r', linestyle='--', alpha=0.7,
                label=f'Train Illicit Rate: {train_illicit_rate:.3f}')
    ax2.set_ylabel('Illicit Rate', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    ax1.set_title('Rolling/Cumulative Performance by Time Step', fontsize=15)
    return pr_fig, temporal_fig


from sklearn.base import BaseEstimator, TransformerMixin

class DropTime(BaseEstimator, TransformerMixin):
    """
    Transformer for dropping the 'time' column from a DataFrame.
    Useful in scikit-learn pipelines.
    """
    def __init__(self, drop=True):
        self.drop=drop
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.drop:
            return X.drop(columns=["time"])
        return X
    
def skorch_splitter(indices, y=None, test_size=0.2, times=None):
    """
    Custom splitter for skorch that performs a temporal split using pre-bound times and test_size.
    Parameters
    ----------
    indices : array-like
        Indices of the data to split (as provided by skorch).
    y : array-like, optional
        Labels (ignored, present for compatibility).
    test_size : float, default=0.2
        Proportion of unique time steps to include in the test split.
    times : array-like
        Array of time values for the full dataset.
    Returns
    -------
    train_indices, test_indices : array-like
        Indices for training and test sets (relative to the input indices).
    """
    return temporal_split(times[indices], test_size=test_size)

def make_skorch_splitter(times, test_size=0.2):
    """
    Factory for skorch-compatible splitter with pre-bound times and test_size.
    Returns a function suitable for use as the train_split argument in skorch.
    Parameters
    ----------
    times : array-like
        Array of time values for the full dataset.
    test_size : float, default=0.2
        Proportion of unique time steps to include in the test split.
    Returns
    -------
    splitter : function
        A function with signature (indices, y=None) that returns train/test indices.
    """
    return partial(skorch_splitter, times=times, test_size=test_size)

class IndexedEcho:
    """
    Dummy indexable object that returns its input index argument.
    Useful for simulating a dataset of a given length where __getitem__ simply echoes the index.
    Can be used for testing or as a placeholder in cross-validation routines.
    """
    def __init__(self, len):
        self.len = len
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return idx