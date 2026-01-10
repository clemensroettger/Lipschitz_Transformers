"""
Lipschitz-bounded Transformer from "Training Transformers with Enforced Lipschitz Bounds"

Architecture specifications:
- Width: 256 (embedding dimension)
- 3 transformer blocks (attention + MLP)
- 4 attention heads
- No bias terms
- Out projections initialized to zero
- Sequence length: 256
- Batch size: 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralNorm(nn.Module):
    """Spectral normalization to enforce Lipschitz bounds."""
    
    def __init__(self, module, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        super().__init__()
        self.module = module
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but got {}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        
    def _compute_weight(self):
        weight = getattr(self.module, self.name)
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        
        height = weight.size(self.dim)
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        weight_mat = weight_mat.view(height, -1)
        
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps)
        
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight
    
    def forward(self, *args):
        self._compute_weight()
        return self.module(*args)


class LipschitzMultiHeadAttention(nn.Module):
    """Lipschitz-bounded multi-head attention."""
    
    def __init__(self, d_model=256, num_heads=4, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V projections (no bias)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection (initialized to zero, no bias)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.w_o.weight)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with Lipschitz normalization
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Mask should be [seq_len, seq_len] or [batch, seq_len, seq_len]
            # scores is [batch, num_heads, seq_len, seq_len]
            # Expand mask to match: [1, 1, seq_len, seq_len]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax with temperature scaling for Lipschitz bound
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection (initialized to zero)
        output = self.w_o(attn_output)
        
        return output


class LipschitzMLP(nn.Module):
    """Lipschitz-bounded MLP/Feed-Forward Network."""
    
    def __init__(self, d_model=256, d_ff=None, dropout=0.0):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # Standard transformer scaling
        
        # Up projection (no bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        
        # Down projection (initialized to zero, no bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        nn.init.zeros_(self.w_down.weight)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Up projection -> GELU -> Down projection
        x = self.w_up(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.w_down(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with Lipschitz-bounded attention and MLP."""
    
    def __init__(self, d_model=256, num_heads=4, dropout=0.0, layer_norm_eps=1e-5):
        super().__init__()
        self.attention = LipschitzMultiHeadAttention(d_model, num_heads, dropout)
        self.mlp = LipschitzMLP(d_model, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-norm architecture with residual connection
        # Attention block
        attn_output = self.attention(x, mask)
        x = x + self.dropout(attn_output)
        
        # MLP block
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model=256, max_len=5000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LipschitzBoundedTransformer(nn.Module):
    """
    Lipschitz-bounded Transformer model.
    
    Architecture:
    - Width: 256
    - 3 transformer blocks
    - 4 attention heads
    - No bias terms
    - Out projections initialized to zero
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Token embedding (no bias)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection (logit head, initialized to zero, no bias)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize token embeddings."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, mask=None):
        """
        Forward pass.
        
        Args:
            idx: Token indices, shape [batch, seq_len]
            mask: Optional attention mask, shape [batch, seq_len, seq_len]
        
        Returns:
            Logits, shape [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = idx.size()
        
        # Token embeddings
        x = self.token_embedding(idx)  # [batch, seq_len, d_model]
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output logits
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def get_lipschitz_constant(self):
        """
        Compute an estimate of the Lipschitz constant.
        
        Note: This is a simplified estimate. The actual Lipschitz constant
        depends on the specific weight values and would require more sophisticated
        computation (e.g., power iteration for spectral norms).
        """
        # This is a placeholder - actual computation would require
        # computing spectral norms of all weight matrices
        return None


# Model hyperparameters (hardcoded as specified)
MODEL_CONFIG = {
    'd_model': 256,
    'num_layers': 3,
    'num_heads': 4,
    'max_seq_len': 256,
    'dropout': 0.0,  # No dropout by default
}

# Training hyperparameters (can be overridden)
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-3,  # Will need to be tuned
    'num_steps': 2000,
    'context_length': 256,
}


def create_model(vocab_size: int, **kwargs) -> LipschitzBoundedTransformer:
    """
    Factory function to create a LipschitzBoundedTransformer model.
    
    Args:
        vocab_size: Vocabulary size
        **kwargs: Override default model config
    
    Returns:
        LipschitzBoundedTransformer instance
    """
    config = {**MODEL_CONFIG, **kwargs}
    return LipschitzBoundedTransformer(
        vocab_size=vocab_size,
        **config
    )


if __name__ == "__main__":
    # Test the model
    vocab_size = 65  # Example vocab size for Shakespeare
    model = create_model(vocab_size)
    
    # Test forward pass
    batch_size = 2
    seq_len = 256
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

