from logging import warn
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import numpy as np
import warnings
from torch_geometric.nn import GAT, GCN

class GNNBinaryClassifier(ClassifierMixin, BaseEstimator):
    """
    Graph Neural Network Binary Classifier with early stopping.
    
    A scikit-learn compatible binary classifier that uses Graph Neural Networks
    for node classification. The classifier automatically stops training when
    the loss stops improving.
    
    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data object containing node features (x), edge indices (edge_index),
        and node labels (y).
    model : torch.nn.Module class
        The GNN model class to instantiate for training.
    hidden_dim : int, default=64
        Number of hidden units in each layer.
    num_layers : int, default=3
        Number of layers in the neural network.
    dropout : float, default=0.5
        Dropout probability for regularization.
    learning_rate_init : float, default=0.01
        Initial learning rate for the Adam optimizer.
    weight_decay : float, default=5e-4
        L2 regularization strength.
    max_iter : int, default=200
        Maximum number of training iterations.
    verbose : bool, default=False
        Whether to print training progress.
    n_iter_no_change : int, default=10
        Number of consecutive iterations with no improvement to trigger early stopping.
    tol : float, default=1e-4
        Tolerance for improvement. Training stops if loss improvement is less than this value.
    device : str or torch.device, default='auto'
        Device to use for computation. Can be 'cpu', 'cuda', 'auto', or a torch.device object.
        If 'auto', will use CUDA if available, otherwise CPU.
    **kwargs : dict
        Additional keyword arguments passed to the model constructor.
        
    Attributes
    ----------
    model_ : torch.nn.Module
        The fitted GNN model.
    loss_curve_ : list
        Training loss at each iteration.
    n_iter_ : int
        Number of iterations run by the solver.
    device_ : torch.device
        The device used for computation.
    """
    
    def _validate_data(self, data):
        """
        Validate that the data object has required attributes.
        
        Parameters
        ----------
        data : object
            Data object to validate.
            
        Returns
        -------
        data : object
            Validated data object.
            
        Raises
        ------
        ValueError
            If data object is missing required attributes.
        """
        attributes = ['x', 'edge_index', 'y']
        for attr in attributes:
            if not hasattr(data, attr):
                raise ValueError(f"Data object must have '{attr}' attribute.")
        return data
    
    def _validate_device(self, device):
        """
        Validate and set the device for computation.
        
        Parameters
        ----------
        device : str or torch.device
            Device specification.
            
        Returns
        -------
        torch.device
            Validated device object.
            
        Raises
        ------
        ValueError
            If device is invalid or CUDA is requested but not available.
        """
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            if device == 'cuda' and not torch.cuda.is_available():
                raise ValueError("CUDA is not available on this system.")
            return torch.device(device)
        elif isinstance(device, torch.device):
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise ValueError("CUDA is not available on this system.")
            return device
        else:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda', 'auto', or a torch.device object.")
    
    def __init__(
        self,
        data,
        model,
        hidden_dim=64,
        num_layers=3,
        dropout=0.5,
        learning_rate_init=0.01,
        weight_decay=5e-4,
        max_iter=200,
        verbose=False,
        n_iter_no_change=10,
        tol=1e-4,
        device='auto',
        heads=None,
        **kwargs,
        ):
        super().__init__()
        self.data = self._validate_data(data)
        self.model = model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate_init = learning_rate_init
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.verbose = verbose
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.device = device  # Store original device parameter
        self.device_ = self._validate_device(device)  # Store validated device
        self.heads = heads
        self.kwargs = kwargs
        
        # Move data to device
        self.data = self.data.to(self.device_)
        
        # Set classes_ attribute for sklearn compatibility
        self.classes_ = np.array([0, 1])
        
        # Handle heads parameter properly
        if heads is not None and model != GAT:
            warnings.warn("'heads' parameter is only applicable for GAT model. Ignoring 'heads'.", UserWarning)
            self.heads = heads  # Store for sklearn but don't use
        elif model == GAT:
            # For GAT, use heads if provided, otherwise default to 1
            actual_heads = heads if heads is not None else 1
            self.heads = actual_heads
            self.kwargs['heads'] = actual_heads
        else:
            # For non-GAT models with heads=None
            self.heads = heads

        if self.verbose:
            print(f"Using device: {self.device_}")

    def _get_pos_weight(self, indices):
        """
        Calculate positive class weight for balanced loss computation.
        
        Parameters
        ----------
        indices : torch.Tensor or array-like
            Indices of training samples.
            
        Returns
        -------
        torch.Tensor
            Weight for positive class to balance the loss.
        """
        y = self.data.y[indices]
        pos_weight = (y == 0).sum() / (y == 1).sum()
        return pos_weight.to(self.device_)

    def fit(self, X, y=None):
        """
        Fit the GNN model to the training data.
        
        Training automatically stops when the loss stops improving for
        n_iter_no_change consecutive iterations, similar to MLPClassifier.
        
        Parameters
        ----------
        train_indices : array-like
            Indices of training samples in the graph.
        y : array-like, default=None
            Target values (ignored, present for sklearn compatibility).
            
        Returns
        -------
        self : GNNBinaryClassifier
            Returns self for method chaining.
            
        Warns
        -----
        UserWarning
            If training stops due to max_iter being reached without convergence.
        """
        train_indices = X
        num_features = self.data.x.shape[1]
        self.model_ = self.model(
            in_channels=num_features,
            hidden_channels=self.hidden_dim,
            out_channels=1,
            num_layers=self.num_layers,
            dropout=self.dropout,
            **self.kwargs
        ).to(self.device_)
        
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.weight_decay)
        
        # Convert indices to tensor and move to device
        if not isinstance(train_indices, torch.Tensor):
            train_indices = torch.tensor(train_indices, dtype=torch.long, device=self.device_)
        else:
            train_indices = train_indices.to(self.device_)
        
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self._get_pos_weight(train_indices))

        # Early stopping variables
        best_loss = float('inf')
        no_improvement_count = 0
        
        self.loss_curve_ = []
        self.n_iter_ = 0

        self.model_.train()
        converged = False
        
        for epoch in range(1, self.max_iter + 1):
            optimizer.zero_grad()
            out = self.model_(self.data.x, self.data.edge_index).squeeze()
            loss = criterion(out[train_indices], self.data.y[train_indices].float())
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            self.loss_curve_.append(current_loss)
            self.n_iter_ = epoch
            
            if self.verbose:
                print(f"Epoch {epoch}: Loss = {current_loss:.6f}")
            
            # Early stopping logic (always enabled)
            if current_loss < best_loss - self.tol:
                best_loss = current_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= self.n_iter_no_change:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}. No improvement for {self.n_iter_no_change} iterations.")
                converged = True
                break
        
        # Warn if training ended without convergence
        if not converged:
            warnings.warn(
                f"Training stopped before reaching convergence. Consider increasing "
                f"max_iter (currently {self.max_iter}) or decreasing tol "
                f"(currently {self.tol}) for better results.",
                UserWarning
            )
        
        # Ensure classes_ is set for sklearn compatibility
        self.classes_ = np.array([0, 1])
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in test_indices.
        
        Parameters
        ----------
        test_indices : array-like
            Indices of test samples in the graph.
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
            
        Raises
        ------
        ValueError
            If the classifier has not been fitted yet.
        """
        test_indices = X
        if not hasattr(self, 'model_'):
            raise ValueError("This GNNBinaryClassifier instance is not fitted yet.")
        
        # Convert indices to tensor and move to device
        if not isinstance(test_indices, torch.Tensor):
            test_indices = torch.tensor(test_indices, dtype=torch.long, device=self.device_)
        else:
            test_indices = test_indices.to(self.device_)
        
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(self.data.x, self.data.edge_index).squeeze()
            predictions = torch.sigmoid(out[test_indices]) > 0.5
            return predictions.int().cpu().numpy()
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in test_indices.
        
        Parameters
        ----------
        test_indices : array-like
            Indices of test samples in the graph.
            
        Returns
        -------
        probabilities : ndarray of shape (n_samples, 2)
            Predicted class probabilities. First column contains probabilities
            for class 0, second column for class 1.
            
        Raises
        ------
        ValueError
            If the classifier has not been fitted yet.
        """
        test_indices = X
        if not hasattr(self, 'model_'):
            raise ValueError("This GNNBinaryClassifier instance is not fitted yet.")
        
        # Convert indices to tensor and move to device
        if not isinstance(test_indices, torch.Tensor):
            test_indices = torch.tensor(test_indices, dtype=torch.long, device=self.device_)
        else:
            test_indices = test_indices.to(self.device_)
        
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(self.data.x, self.data.edge_index).squeeze()
            # Debug: Check for inf/nan in raw outputs
            raw_outputs = out[test_indices]
            if torch.isnan(raw_outputs).any() or torch.isinf(raw_outputs).any():
                warnings.warn("Model outputs contain NaN or Inf values. Using fallback predictions.")
                # Fallback to neutral probabilities
                proba_positive = np.full(len(test_indices), 0.5)
            else:
                # Clamp extreme values to prevent numerical issues
                raw_outputs = torch.clamp(raw_outputs, min=-10, max=10)
                proba_positive = torch.sigmoid(raw_outputs).cpu().numpy()
            
            proba_negative = 1 - proba_positive
            return np.column_stack([proba_negative, proba_positive])
    
    # def decision_function(self, X):
    #     """
    #     Predict confidence scores for samples.
        
    #     Parameters
    #     ----------
    #     X : array-like
    #         Indices of test samples in the graph.
            
    #     Returns
    #     -------
    #     scores : ndarray of shape (n_samples,)
    #         Confidence scores per sample for the positive class.
    #     """
    #     test_indices = X
    #     if not hasattr(self, 'model_'):
    #         raise ValueError("This GNNBinaryClassifier instance is not fitted yet.")
        
    #     # Convert indices to tensor and move to device
    #     if not isinstance(test_indices, torch.Tensor):
    #         test_indices = torch.tensor(test_indices, dtype=torch.long, device=self.device_)
    #     else:
    #         test_indices = test_indices.to(self.device_)
        
    #     self.model_.eval()
    #     with torch.no_grad():
    #         out = self.model_(self.data.x, self.data.edge_index).squeeze()
    #         return out[test_indices].cpu().numpy()