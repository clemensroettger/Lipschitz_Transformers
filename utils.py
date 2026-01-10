import torch
import torch.nn as nn

class SpectralConstraint(object):
    """
    Applies the 'Projected Spectral Normalization' constraint from the paper.
    This should be called AFTER the optimizer.step().
    """
    def __init__(self, model: nn.Module, sigma_max: float = 1.0, n_power_iterations: int = 2, eps: float = 1e-12):
        self.sigma_max = sigma_max
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.modules_to_constrain = []
        
        # Register hooks or buffers for persistent u/v vectors
        # The paper applies this to "every linear layer weight matrix" 
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._register_buffers(module)
                self.modules_to_constrain.append(module)
    
    def _register_buffers(self, module: nn.Linear):
        """
        Initialize persistent u and v vectors for power iteration.
        """
        weight = module.weight
        height, width = weight.shape
        
        # We need u (height) and v (width)
        # Check if they already exist to avoid resetting them
        if not hasattr(module, 'spectral_u'):
            u = torch.randn(height, device=weight.device, dtype=weight.dtype)
            module.register_buffer('spectral_u', nn.functional.normalize(u, dim=0))
            
        if not hasattr(module, 'spectral_v'):
            v = torch.randn(width, device=weight.device, dtype=weight.dtype)
            module.register_buffer('spectral_v', nn.functional.normalize(v, dim=0))

    @torch.no_grad()
    def step(self):
        """
        Performs the weight projection. Call this after optimizer.step().
        """
        for module in self.modules_to_constrain:
            weight = module.weight
            u = module.spectral_u
            v = module.spectral_v
            
            # --- 1. Power Iteration (2 iterations as per paper) ---
            for _ in range(self.n_power_iterations):
                # v = normalized(W^T @ u)
                v = torch.mv(weight.t(), u)
                v = nn.functional.normalize(v, dim=0, eps=self.eps)
                
                # u = normalized(W @ v)
                u = torch.mv(weight, v)
                u = nn.functional.normalize(u, dim=0, eps=self.eps)
            
            # Update the buffers for the next step
            module.spectral_u.copy_(u)
            module.spectral_v.copy_(v)
            
            # --- 2. Estimate Sigma_1 ---
            # sigma_1 = u^T @ W @ v
            sigma_1 = torch.dot(u, torch.mv(weight, v))
            
            # --- 3. Apply Projection (Constraint) ---
            # "apply the mapping W -> (sigma_max / max(sigma_1, sigma_max)) * W"
            if sigma_1 > self.sigma_max:
                scale = self.sigma_max / sigma_1
                module.weight.mul_(scale)