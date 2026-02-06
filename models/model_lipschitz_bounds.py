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



class LipschitzMultiHeadAttention(nn.Module):
    """Lipschitz-bounded multi-head attention."""
    
    def __init__(self, d_model=256, num_heads=4, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V projections (no bias) - orthogonal init for spectral norm = 1
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        nn.init.orthogonal_(self.w_q.weight)
        nn.init.orthogonal_(self.w_k.weight)
        nn.init.orthogonal_(self.w_v.weight)
        
        # Output projection - orthogonal init for spectral norm = 1
        # (Paper: "At initialization, we project the linear weights to be semi-orthogonal")
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        nn.init.orthogonal_(self.w_o.weight)
        
        self.dropout = nn.Dropout(dropout)

        # divide by d_k instead of sqrt(d_k) to make attention 1-Lipschitz if every input has unit norm
        self.scale = 1.0 / self.d_k
        
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
        
        # Output projection (initialized to zero), multiply by 1/3 to make attention 1-Lipschitz
        output = 1/3 * self.w_o(attn_output)
        
        return output


class LipschitzMLP(nn.Module):
    """Lipschitz-bounded MLP/Feed-Forward Network."""
    
    def __init__(self, d_model=256, d_ff=None, dropout=0.0):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # Standard transformer scaling
        
        # Up projection - orthogonal (semi-orthogonal for non-square) init for spectral norm = 1
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        nn.init.orthogonal_(self.w_up.weight)
        
        # Down projection - orthogonal (semi-orthogonal for non-square) init for spectral norm = 1
        # (Paper: "At initialization, we project the linear weights to be semi-orthogonal")
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        nn.init.orthogonal_(self.w_down.weight)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Up projection -> GELU -> Down projection
        x = self.w_up(x)
        # divide gelu by maximum derivative to make it 1-Lipschitz
        x = F.gelu(x)/1.1289
        x = self.dropout(x)
        x = self.w_down(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with Lipschitz-bounded attention and MLP."""
    
    def __init__(self, d_model=256, num_heads=4, dropout=0.0, num_res_conn = 6, layer_norm_eps=1e-5):
        super().__init__()
        self.attention = LipschitzMultiHeadAttention(d_model, num_heads, dropout)
        self.mlp = LipschitzMLP(d_model, dropout=dropout)
        self.N = num_res_conn
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):

        
        # Pre-norm architecture with residual connection
        # Attention block
        attn_output = self.attention(x, mask)

        # reparameterize the residual connection
        x = (self.N-1)/self.N * x + 1/self.N * self.dropout(attn_output)
        
        # MLP block
        mlp_output = self.mlp(x)

        # reparameterize the residual connection
        x = (self.N-1)/self.N * x + 1/self.N * self.dropout(mlp_output)
        
        return x


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding with spectral norm normalization for Lipschitz bounds.
    
    Each position has a learnable embedding vector that is normalized to unit 
    spectral (L2) norm. Combined with normalized token embeddings using a convex
    combination to maintain ||x_0||_{∞,2} ≤ 1.
    
    Args:
        d_model: Embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        pe_weight: Weight for positional encoding in convex combination (default: 0.5)
                   x_out = (1 - pe_weight) * token_embed + pe_weight * pos_embed
    """
    
    def __init__(self, d_model=256, max_seq_len=256, dropout=0.0, pe_weight=0.5):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe_weight = pe_weight
        self.dropout = nn.Dropout(dropout)
        
        # Learned positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Initialize with small random values
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        # Normalize to unit spectral norm
        self.normalize_positional_embeddings()
    
    @torch.no_grad()
    def normalize_positional_embeddings(self):
        """
        Normalize positional embedding rows to have unit spectral (L2) norm.
        
        This should be called after each optimizer step during training
        to maintain the Lipschitz bound assumption.
        """
        pos_weight = self.pos_embedding.weight
        # Compute L2 norm of each row
        l2_norm = pos_weight.pow(2).sum(dim=1, keepdim=True).sqrt()
        # Avoid division by zero
        l2_norm = l2_norm.clamp(min=1e-8)
        # Normalize to unit spectral norm
        pos_weight.div_(l2_norm)
        
    def forward(self, x):
        """
        Apply positional encoding using convex combination.
        
        Args:
            x: Token embeddings, shape [batch, seq_len, d_model]
            
        Returns:
            Combined embeddings with positional information, shape [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # Get position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        
        # Get positional embeddings
        pos_embed = self.pos_embedding(positions)  # [1, seq_len, d_model]
        
        # Convex combination to preserve spectral norm bound:
        # ||αa + (1-α)b||_2 ≤ α||a||_2 + (1-α)||b||_2 ≤ 1 when ||a||_2, ||b||_2 ≤ 1
        x = (1 - self.pe_weight) * x + self.pe_weight * pos_embed
        
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
        
        # Learned positional encoding
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout, num_res_conn=2*num_layers)
            for _ in range(num_layers)
        ])
        
        # Output projection (logit head) - orthogonal init for spectral norm = 1
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.orthogonal_(self.lm_head.weight)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize token and positional embeddings with unit spectral norm."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        # Normalize each embedding row to have unit spectral norm
        self.normalize_embeddings()
    
    @torch.no_grad()
    def normalize_embeddings(self):
        """
        Normalize token and positional embedding rows to have unit spectral (L2) norm.
        
        This should be called after each optimizer step during training
        to maintain the Lipschitz bound assumption that ||x_0||_{∞,2} ≤ 1.
        
        Both token embeddings and positional embeddings are normalized to unit
        spectral norm, then combined using a convex combination in the forward pass.
        
        Note: The embedding weights have shape [num_embeddings, d_model], where each
        row is an embedding vector. We normalize each row to unit spectral (L2) norm.
        """
        # Normalize token embeddings
        embed_weight = self.token_embedding.weight
        l2_norm = embed_weight.pow(2).sum(dim=1, keepdim=True).sqrt()
        l2_norm = l2_norm.clamp(min=1e-8)
        embed_weight.div_(l2_norm)
        
        # Normalize positional embeddings
        self.pos_encoding.normalize_positional_embeddings()
        
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


def compute_spectral_norm(weight: torch.Tensor) -> float:
    """
    Compute the l2 operator norm of a weight matrix.

    Args:
        weight: Weight matrix of shape [d_out, d_in]
    
    Returns:
        The spectral norm of weight matrix
    """
    spectral_norm = torch.linalg.matrix_norm(weight, ord=2).item()
    return spectral_norm


def compute_embedding_spectral_norms(model: LipschitzBoundedTransformer) -> dict:
    """
    Compute the spectral norms of token and positional embeddings.
    
    Args:
        model: A LipschitzBoundedTransformer instance
    
    Returns:
        Dictionary with max, min, and mean spectral norms of embedding rows
        for both token and positional embeddings
    """
    # Token embeddings
    token_weight = model.token_embedding.weight
    token_spectral_norms = token_weight.pow(2).sum(dim=1).sqrt()
    
    # Positional embeddings
    pos_weight = model.pos_encoding.pos_embedding.weight
    pos_spectral_norms = pos_weight.pow(2).sum(dim=1).sqrt()
    
    return {
        'token_max_spectral': token_spectral_norms.max().item(),
        'token_min_spectral': token_spectral_norms.min().item(),
        'token_mean_spectral': token_spectral_norms.mean().item(),
        'pos_max_spectral': pos_spectral_norms.max().item(),
        'pos_min_spectral': pos_spectral_norms.min().item(),
        'pos_mean_spectral': pos_spectral_norms.mean().item(),
    }


def compute_lipschitz_certificate(model: LipschitzBoundedTransformer) -> dict:
    r"""
    Compute the Lipschitz upper bound with regards to the matrix max spectral norm of a transformer 

    The max spectral norm over token positions for a matrix W of shape seq_len x d_model 
    is defined as ||W||_{\infty,2} := max_{t \in 1, \dots T} ||W_t||_2.

    
    This implements the algorithm from Appendix C of 
    "Training Transformers with Enforced Lipschitz Bounds" (arXiv:2507.13338).
    
    The algorithm proceeds in two steps:
    1. Bound activation norms everywhere in the network (Step 4)
    2. Bound the Lipschitz constant using Steps 1-3
    
    Note: This assumes embeddings are normalized to unit spectral norm.
    Call model.normalize_embeddings() before computing the certificate
    if embeddings may have drifted during training.
    
    Args:
        model: A LipschitzBoundedTransformer instance
    
    Returns:
        Dictionary containing:
        - 'lipschitz_bound': The global Lipschitz upper bound
        - 'activation_bounds': List of activation norm bounds at each layer
        - 'block_lipschitz_bounds': Lipschitz bounds for each block
        - 'layer_lipschitz_bounds': Cumulative Lipschitz bound after each layer
    """
    GELU_MAX_DERIVATIVE = 1.1289
    
    # Get model parameters
    num_layers = model.num_layers
    num_residual_connections = 2 * num_layers  # attention + MLP per block
    alpha = 1.0 / num_residual_connections  # residual connection weight
    
    # =========================================================================
    # Step 4: Bound activation norms through the network
    # =========================================================================
    # Start with embedding having spectral norm ≤ 1 (assumed normalized)
    # Note: The paper assumes embeddings are spectral normalized to 1
    activation_bounds = [1.0]  # ||x_0||_{∞,2} ≤ 1
    
    block_info = []  # Store detailed info for each block
    
    for layer_idx, block in enumerate(model.blocks):
        current_activation_bound = activation_bounds[-1]
        
        # --- Attention block ---
        attn = block.attention
        
        # Get weight norms for attention
        w_q_norm = compute_spectral_norm(attn.w_q.weight)
        w_k_norm = compute_spectral_norm(attn.w_k.weight)
        w_v_norm = compute_spectral_norm(attn.w_v.weight)
        w_o_norm = compute_spectral_norm(attn.w_o.weight)
        
        # Attention can increase activation norm by ||W_out|| · ||W_v|| · ||x_i||
        # (See Step 4 in paper, equation 24)
        # note the 1/3 scaling on attention output
        attn_output_norm = w_o_norm * w_v_norm * current_activation_bound
        scaled_attn_output_norm = (1/3) * attn_output_norm

        # Apply residual connection: (1-α)||x_i|| + α||block(x_i)||
        after_attn_residual = (1 - alpha) * current_activation_bound + alpha * scaled_attn_output_norm
        activation_bounds.append(after_attn_residual)
        
        # --- MLP block ---
        mlp = block.mlp
        
        w_up_norm = compute_spectral_norm(mlp.w_up.weight)
        w_down_norm = compute_spectral_norm(mlp.w_down.weight)
        
        # MLP can increase activation norm by ||W_down|| · ||W_up|| / 1.1289
        # (since |GeLU(x)| ≤ |x| element-wise)
        mlp_output_norm = (w_down_norm * w_up_norm / GELU_MAX_DERIVATIVE) * after_attn_residual
        
        # Apply residual connection
        after_mlp_residual = (1 - alpha) * after_attn_residual + alpha * mlp_output_norm
        
        activation_bounds.append(after_mlp_residual)
        
        # Store block info
        block_info.append({
            'layer': layer_idx,
            'w_q_norm': w_q_norm,
            'w_k_norm': w_k_norm,
            'w_v_norm': w_v_norm,
            'w_o_norm': w_o_norm,
            'w_up_norm': w_up_norm,
            'w_down_norm': w_down_norm,
            'activation_after_attn': after_attn_residual,
            'activation_after_mlp': after_mlp_residual,
        })
    
    # =========================================================================
    # Steps 1-3: Bound the Lipschitz constant
    # =========================================================================
    
    # Initialize: embedding layer is 1-Lipschitz (assuming normalized embeddings)
    lipschitz_bound = 1.0
    layer_lipschitz_bounds = [1.0]
    block_lipschitz_bounds = []
    
    # Track activation norm at each point for attention Lipschitz calculation
    current_act_idx = 0  # Index into activation_bounds
    
    for layer_idx, block in enumerate(model.blocks):
        info = block_info[layer_idx]
        current_activation_bound = activation_bounds[current_act_idx]
        
        # --- Step 3: Attention Lipschitz bound ---
        attn = block.attention
        
        # Query, key, value norms at this point in the network
        # ||q|| = ||W_Q|| · ||x||, ||k|| = ||W_K|| · ||x||, ||v|| = ||W_V|| · ||x||
        q_norm = info['w_q_norm'] * current_activation_bound
        k_norm = info['w_k_norm'] * current_activation_bound
        v_norm = info['w_v_norm'] * current_activation_bound
        
        # Theorem C.1: Functional attention has Lipschitz bound
        # L_F = max(1, ||v|| · max(||q||, ||k||))
        L_functional_attn = max(1.0, v_norm * max(q_norm, k_norm))
        
        # For attention scale s_attn ≠ 1/d_Q, multiply by √(s_attn · d_Q)
        # In our model: scale = 1/d_k where d_k = d_model/num_heads
        # Since d_k = d_Q (head dimension), √(s_attn · d_Q) = √((1/d_k) · d_k) = 1
        # So no additional factor needed
        
        # Full attention block: (1/3) · W_out ∘ F
        # The 1/3 comes from the 3-sensitivity of the (q,k,v) tuple
        # L_attn = (1/3) · ||W_out||_{RMS→RMS} · L_F
        L_attn_block = (1/3) * info['w_o_norm'] * L_functional_attn
        
        # --- Step 1: Residual connection for attention ---
        # After residual: (1-α)·L + α·L·L_block
        lipschitz_bound = (1 - alpha) * lipschitz_bound + alpha * lipschitz_bound * L_attn_block
        layer_lipschitz_bounds.append(lipschitz_bound)
        block_lipschitz_bounds.append(('attn', layer_idx, L_attn_block))
        
        current_act_idx += 1
        current_activation_bound = activation_bounds[current_act_idx]
        
        # --- Step 2: MLP Lipschitz bound ---
        # L_MLP = ||W_down||_{2} · ||W_up||_{2}
        L_mlp_block = info['w_down_norm'] * info['w_up_norm']
        
        # --- Step 1: Residual connection for MLP ---
        lipschitz_bound = (1 - alpha) * lipschitz_bound + alpha * lipschitz_bound * L_mlp_block
        layer_lipschitz_bounds.append(lipschitz_bound)
        block_lipschitz_bounds.append(('mlp', layer_idx, L_mlp_block))
        
        current_act_idx += 1
    
    # --- Final output projection (lm_head) ---
    lm_head_norm = compute_spectral_norm(model.lm_head.weight)
    final_lipschitz_bound = lipschitz_bound * lm_head_norm
    
    # --- Verify embedding normalization ---
    embedding_norms = compute_embedding_spectral_norms(model)
    
    return {
        'lipschitz_bound': final_lipschitz_bound,
        'lipschitz_before_head': lipschitz_bound,
        'lm_head_norm': lm_head_norm,
        'activation_bounds': activation_bounds,
        'block_lipschitz_bounds': block_lipschitz_bounds,
        'layer_lipschitz_bounds': layer_lipschitz_bounds,
        'block_info': block_info,
        'num_residual_connections': num_residual_connections,
        'alpha': alpha,
        'embedding_norms': embedding_norms,
    }


def print_lipschitz_report(result: dict):
    """Print a formatted report of the Lipschitz certificate computation."""
    print("=" * 60)
    print("Lipschitz Certificate Report")
    print("=" * 60)
    
    print(f"\nGlobal Lipschitz Upper Bound: {result['lipschitz_bound']:.6e}")
    print(f"Lipschitz Bound (before lm_head): {result['lipschitz_before_head']:.6e}")
    print(f"LM Head Norm: {result['lm_head_norm']:.6f}")
    
    print(f"\nResidual connection α = 1/{result['num_residual_connections']} = {result['alpha']:.6f}")
    
    # Embedding normalization verification
    embed_norms = result['embedding_norms']
    print("\n" + "-" * 60)
    print("Embedding Normalization Verification:")
    print("-" * 60)
    print("  Token Embeddings:")
    print(f"    Max spectral norm:  {embed_norms['token_max_spectral']:.6f}")
    print(f"    Min spectral norm:  {embed_norms['token_min_spectral']:.6f}")
    print(f"    Mean spectral norm: {embed_norms['token_mean_spectral']:.6f}")
    if abs(embed_norms['token_max_spectral'] - 1.0) < 1e-5:
        print("    ✓ Token embeddings properly normalized")
    else:
        print(f"    ⚠ Warning: Token embeddings not normalized! Max = {embed_norms['token_max_spectral']:.6f}")
    
    print("  Positional Embeddings:")
    print(f"    Max spectral norm:  {embed_norms['pos_max_spectral']:.6f}")
    print(f"    Min spectral norm:  {embed_norms['pos_min_spectral']:.6f}")
    print(f"    Mean spectral norm: {embed_norms['pos_mean_spectral']:.6f}")
    if abs(embed_norms['pos_max_spectral'] - 1.0) < 1e-5:
        print("    ✓ Positional embeddings properly normalized")
    else:
        print(f"    ⚠ Warning: Positional embeddings not normalized! Max = {embed_norms['pos_max_spectral']:.6f}")
    
    print("\n" + "-" * 60)
    print("Activation Norm Bounds (||·||_{∞,2}):")
    print("-" * 60)
    for i, bound in enumerate(result['activation_bounds']):
        if i == 0:
            print(f"  After embedding:      {bound:.6f}")
        else:
            block_idx = (i - 1) // 2
            is_attn = (i - 1) % 2 == 0
            block_type = "attention" if is_attn else "MLP"
            print(f"  After block {block_idx} {block_type:9s}: {bound:.6f}")
    
    print("\n" + "-" * 60)
    print("Block Lipschitz Bounds:")
    print("-" * 60)
    for block_type, layer_idx, L_block in result['block_lipschitz_bounds']:
        print(f"  Block {layer_idx} {block_type:4s}: {L_block:.6f}")
    
    print("\n" + "-" * 60)
    print("Cumulative Lipschitz Bounds:")
    print("-" * 60)
    for i, L in enumerate(result['layer_lipschitz_bounds']):
        if i == 0:
            print(f"  After embedding:      {L:.6e}")
        else:
            block_idx = (i - 1) // 2
            is_attn = (i - 1) % 2 == 0
            block_type = "attention" if is_attn else "MLP"
            print(f"  After block {block_idx} {block_type:9s}: {L:.6e}")
    
    print("\n" + "-" * 60)
    print("Weight Norms (||·||_{2}):")
    print("-" * 60)
    for info in result['block_info']:
        layer = info['layer']
        print(f"  Block {layer}:")
        print(f"    W_Q: {info['w_q_norm']:.6f}, W_K: {info['w_k_norm']:.6f}, "
              f"W_V: {info['w_v_norm']:.6f}, W_O: {info['w_o_norm']:.6f}")
        print(f"    W_up: {info['w_up_norm']:.6f}, W_down: {info['w_down_norm']:.6f}")
    
    print("=" * 60)


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
    
    # Compute and print Lipschitz certificate
    print("\n")
    result = compute_lipschitz_certificate(model)
    print_lipschitz_report(result)

