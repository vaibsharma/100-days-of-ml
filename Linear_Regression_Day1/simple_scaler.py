import numpy as np
import torch

class SimpleStandardScaler:
    """Simple StandardScaler implementation for M1 Mac compatibility"""
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        """Compute the mean and std to be used for later scaling"""
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        
        # Handle constant features (std = 0)
        self.scale_[self.scale_ == 0] = 1.0
        
        return self
    
    def transform(self, X):
        """Scale the data"""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before transform")
        
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
            result = (X_np - self.mean_) / self.scale_
            return torch.FloatTensor(result)
        else:
            return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit to data, then transform it"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Scale back the data to the original representation"""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        if isinstance(X, torch.Tensor):
            X_np = X.numpy()
            result = X_np * self.scale_ + self.mean_
            return torch.FloatTensor(result)
        else:
            return X * self.scale_ + self.mean_ 