"""
Muon optimizer implementation.

Based on "Training Transformers with Enforced Lipschitz Bounds" (Jordan et al., 2024).

Muon orthogonalizes gradient updates using Newton-Schulz iteration, ensuring
updates have bounded spectral norm. This helps maintain Lipschitz bounds during training.

Key properties:
- Gradient updates are orthogonalized for 2D weight matrices
- Update spectral norm is bounded (close to 1 after orthogonalization)
- Uses momentum with optional Nesterov acceleration
"""

import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    """
    Muon optimizer with Newton-Schulz orthogonalization.
    
    Applies orthogonalization to gradient updates for 2D weight matrices,
    ensuring the update has a known, bounded spectral norm.
    
    Args:
        params: Iterable of parameters or param groups
        lr: Learning rate (default: 0.02)
        momentum: Momentum factor (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
        
        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer on first step
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                buf = state['momentum_buffer']
                
                # Update momentum buffer: buf = momentum * buf + grad
                buf.mul_(momentum).add_(grad)
                
                # Compute update direction
                if nesterov:
                    # Nesterov: look ahead by using grad + momentum * buf
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf.clone()
                
                # For 2D weight matrices, apply Newton-Schulz orthogonalization
                if update.dim() == 2:
                    update = self._newton_schulz_orthogonalize(update, ns_steps)
                
                # Apply the update: p = p - lr * update
                p.add_(update, alpha=-lr)
        
        return loss
    
    def _newton_schulz_orthogonalize(self, G, ns_steps):
        """
        Orthogonalize matrix G using Newton-Schulz iteration.
        
        Uses the quintic polynomial approximation for faster convergence.
        This ensures the update has spectral norm close to 1, which helps
        maintain Lipschitz bounds during training.
        
        Args:
            G: Gradient matrix to orthogonalize [out_features, in_features]
            ns_steps: Number of Newton-Schulz iterations
        
        Returns:
            Orthogonalized matrix with bounded spectral norm
        """
        assert G.dim() == 2
        
        # Optimized coefficients for quintic Newton-Schulz (faster convergence)
        a, b, c = (3.4445, -4.7750, 2.0315)
        
        # Normalize by Frobenius norm
        norm = G.norm()
        if norm < 1e-8:
            return G
        
        X = G / norm
        
        # Transpose if more rows than columns (work with smaller dimension)
        transpose = G.size(0) > G.size(1)
        if transpose:
            X = X.T
        
        # Newton-Schulz iteration with quintic polynomial
        # X = a*X + b*(X @ X.T @ X) + c*(X @ X.T @ X @ X.T @ X)
        for _ in range(ns_steps):
            A = X @ X.T
            B = A @ X
            X = a * X + b * B + c * (A @ B)
        
        if transpose:
            X = X.T
        
        return X


def normalize_embedding_columns(grad, max_inflation=16.0):
    """
    Normalize embedding gradient columns with capped inflation factor.
    
    From the paper: "normalizing the columns of embedding gradient,
    as suggested by the ℓ1 → RMS duality map (Bernstein and Newhouse, 2025)"
    
    "We choose to maximally multiply each column by 16 during the dualization step."
    
    Args:
        grad: Embedding gradient tensor [vocab_size, d_model]
        max_inflation: Maximum inflation factor per column (default: 16.0)
    
    Returns:
        Normalized gradient tensor
    """
    if grad is None:
        return None
    
    # Compute L2 norm of each column
    col_norms = torch.norm(grad, dim=0, keepdim=True)  # [1, d_model]
    
    # Target RMS scale (for ℓ1 → RMS duality)
    rms_target = grad.shape[0] ** 0.5
    
    # Compute inflation factors, capped at max_inflation
    # This normalizes columns to have similar scale, but limits extreme inflation
    inflation = torch.clamp(rms_target / (col_norms + 1e-8), max=max_inflation)
    
    return grad * inflation
