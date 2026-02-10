"""
Transformer archtiecture with L2-attention

Architecture specifications:
- Width: 256 (embedding dimension)
- 3 transformer blocks (attention + MLP)
- 4 attention heads
- No bias terms
- Out projections initialized to zero
- Sequence length: 256
- Batch size: 64
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        nn.init.normal_(self.pos_encoding.weight, mean=0, std=d_model**-0.5)

    def forward(self, input):
        """
        Forward Pass.

        Args:
            input: Embedded sequence, shape [batch, seq_len, d_model]

        Returns:
            output: Embedded sequence with positional encodings, shape [batch, seq_len, d_model]
        """

        seq_len = input.size()[1]

        idx = torch.arange(0, seq_len, device=input.device).unsqueeze(0)

        pos_encoding = self.pos_encoding(idx)

        return input + pos_encoding


class LipschitzMLP(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        # in projection
        self.w_in = nn.Linear(d_model, 4*d_model, bias=False)

        # out projection, initialized to zero (as in https://arxiv.org/pdf/2507.13338)
        self.w_out = nn.Linear(4*d_model, d_model, bias=False)
        nn.init.zeros_(self.w_out.weight)
    
    def forward(self, x):
        x = self.w_in(x)
        # Gelu devided by maximum derivative to make it 1-Lipschitz
        x = F.gelu(x)/1.1289
        x = self.w_out(x)

        return x


class L2MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # for L2 Attention to be Lipschitz we need W_k=W_q
        # instead of using a seperate linear projection for every head, 
        # we have one d_modelxd_model tensor to project all heads at one
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        # Value projection (needed for merged transform computation)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.output_projection = nn.Linear(d_model, d_model, bias=False)


    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # compute query and keys and view/transpose to get get shape (batch_size, num_heads, seq_len, d_k)
        Q = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Shape (batch_size, num_heads, seq_len, 1)
        query_l2_row_norm = torch.linalg.vector_norm(Q, ord=2, dim=-1, keepdim=True)**2
        

        # Shape (batch_size, num_heads, 1, seq_len)
        keys_l2_row_norm = (torch.linalg.vector_norm(K, ord=2, dim=-1, keepdim=True)**2).transpose(-2, -1)

        dot_product = torch.matmul(Q, K.transpose(-2, -1))

        squared_dist = query_l2_row_norm  - 2 * dot_product + keys_l2_row_norm

        # Scale and negate
        scores = -squared_dist/torch.sqrt(torch.tensor(self.d_k, dtype=x.dtype, device=x.device))

        # Apply mask if provided (FIX: Added mask support)
        if mask is not None:
            # Mask should be [seq_len, seq_len] or [batch, seq_len, seq_len]
            # scores is [batch, num_heads, seq_len, seq_len]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Shape (batch_size, num_head, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Instead of computing P^h * X * A_h, we compute (P^h W^Q) ( (W^Q)^T W^V)/sqrt(d_k) 
        # Shape (batch_size, num_head, seq_len, d_k)
        context_q = torch.matmul(attn_weights, Q)

        # transposed weights
        # Since W_k = W_q, use w_k for w_q
        # Reshape from (d_model, d_model) to (num_heads, d_k, d_model)
        # Make weights contiguous before reshaping
        # Weight matrix is (d_model, d_model) = (out_features, in_features)
        # We need to reshape to (num_heads, d_k, d_model) where d_k = d_model // num_heads
        # Total elements: num_heads * d_k * d_model = d_model * d_model ✓
        w_k_contiguous = self.w_k.weight.contiguous()
        w_v_contiguous = self.w_v.weight.contiguous()
        
        # Validate dimensions before reshape
        expected_elements = self.num_heads * self.d_k * self.d_model
        actual_elements = w_k_contiguous.numel()
        if expected_elements != actual_elements:
            raise ValueError(f"Weight reshape dimension mismatch: expected {expected_elements} elements, got {actual_elements}. d_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}")
        
        # Reshape to (num_heads, d_k, d_model)
        w_q_params = w_k_contiguous.view(self.num_heads, self.d_k, self.d_model)
        w_v_params = w_v_contiguous.view(self.num_heads, self.d_k, self.d_model)

        # want to compute  (W^Q)^T @ W^V
        # output shaoe: (Heads, d_k, d_k)
        # h=heads, k=d_k (in), v=d_k (out), d=d_model
        merged_transform = torch.einsum('hkd, hvd -> hkv', w_q_params, w_v_params)

        # Combine context_q and merged_transform to get output of L2 self-attention P^h * X * A_h = context_q
        # Output: (Batch, Heads, Seq, D_k)
        attn_output = torch.einsum('bhsk, hkv -> bhsv', context_q, merged_transform)

        # Step D: Apply the scaling factor from A_h definition
        # A_h includes division by sqrt(D/H)
        attn_output = attn_output / torch.sqrt(torch.tensor(self.d_k, dtype=x.dtype, device=x.device))

        # --- 4. Final Projection ---
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Apply W^O
        output = self.output_projection(attn_output)
        
        return output



class L2TransformerBlock(nn.Module):


    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, num_res_conn: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        # N is the number of residual connections in this block (attention + MLP = 2)
        self.N = num_res_conn

        self.mlp = LipschitzMLP(d_model)
        self.L2MHA = L2MultiHeadAttention(d_model, num_heads)

    def forward(self, x, mask=None):
        # L2 MultiHeadAttention
        attn_output = self.L2MHA(x, mask)

        # reparameterized residual connection
        x = (self.N-1)/self.N * x + 1/self.N * attn_output

        mlp_output = self.mlp(x)

        # reparamerized residual connection
        output = (self.N-1)/self.N * x + 1/self.N * mlp_output

        return output




class L2Transformer(nn.Module):
    """
    Transformer model with L2 self-attention mechanism (see https://arxiv.org/pdf/2006.04710)

    Architecture specifications:
    - Width: 256 (embedding dimension)
    - 3 transformer blocks (L2 self-attention + MLP)
    - 4 attention heads
    - No bias terms
    - Out projections initialized to zero
    """

    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 3, num_heads: int = 4, max_seq_len: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len)

        self.blocks = nn.ModuleList([
            L2TransformerBlock(d_model, num_heads, max_seq_len, 2*num_layers) 
            for _ in range(num_layers)
        ])

        self.logit_layer = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.zeros_(self.logit_layer.weight)

        self._init_token_weights()
        
    def _init_token_weights(self):
        """Initialize token embeddings"""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

    def forward(self, input_tokens, mask=None):
        """
        Forward pass.

        Args:
            input_tokens: indices of input_tokens, shape [batch, seq_len]
            mask: Optional attention mask, shape [batch, seq_len, seq_len]

        Returns:
            Logits, shape [batch, seq_len, vocab_size]
        """

        # Token embeddings
        x = self.token_embedding(input_tokens)

        # Apply positional encoding to embedding
        x = self.pos_encoding(x)

        # num_layer of Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)

        # final logits
        logits = self.logit_layer(x)

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

def compute_l2_lipschitz_certificate(model: L2Transformer) -> dict:
    r"""
    Compute the Lipschitz upper bound with regards to the spectral norm.
    
    As L2 Self-Attention is Lipschitz continuous it is sufficient to upper bound the 
    Lipschitz constant of every layer and multiply the layerwise constants.
    
    Args:
        model: L2Transformer instance
    
    Returns:
        Dictionary containing:
        - 'lipschitz_bound': The global Lipschitz upper bound
        - 'activation_bounds': List of activation norm bounds at each layer
        - 'block_lipschitz_bounds': Lipschitz bounds for each block
        - 'layer_lipschitz_bounds': Cumulative Lipschitz bound after each layer
    """
    pass



# Model hyperparameters (default config)
MODEL_CONFIG = {
    'd_model': 256,
    'num_layers': 3,
    'num_heads': 4,
    'max_seq_len': 256,
}

# Training hyperparameters (can be overridden)
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'num_steps': 2000,
    'context_length': 256,
}


def create_model(vocab_size: int, **kwargs) -> L2Transformer:
    """
    Factory function to create an L2Transformer model.
    
    Args:
        vocab_size: Vocabulary size
        **kwargs: Override default model config
    
    Returns:
        L2Transformer instance
    """
    config = {**MODEL_CONFIG, **kwargs}
    return L2Transformer(
        vocab_size=vocab_size,
        **config
    )


if __name__ == "__main__":
    # Test the model
    vocab_size = 65
    model = create_model(vocab_size)
    
    batch_size = 2
    seq_len = 256
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(x)
    print(f"Success: Input shape: {x.shape}, Output logits shape: {logits.shape}")

