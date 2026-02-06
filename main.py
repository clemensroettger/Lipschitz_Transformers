"""
Training script for LipschitzBoundedTransformer on Shakespeare dataset.

Uses Muon optimizer as described in "Training Transformers with Enforced Lipschitz Bounds":
- Muon for all linear layer weight matrices (including logit head)
- Column-normalized gradients for embeddings with max inflation factor 16
- Per-layer learning rate decay (1/2 per residual layer)
- blocks_mass = 32 (embeddings learn at base_lr / 32)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from load_shakespeare import load_shakespeare
from models import create_model, LIPSCHITZ_BOUNDED_TRAINING_CONFIG, L2_TRAINING_CONFIG, compute_lipschitz_certificate, print_lipschitz_report
from utils import SpectralConstraint
from muon import Muon, normalize_embedding_columns


def get_parameter_groups(model, base_lr, blocks_mass=32.0, num_layers=3):
    """
    Create parameter groups for training with Muon.
    
    From paper Appendix F:
    - blocks_mass = 32: embeddings learn at base_lr / 32
    - Per-layer LR decay: factor of 1/2 per residual layer
      (later layers train more than earlier layers)
    
    Args:
        model: The transformer model
        base_lr: Base learning rate for Muon
        blocks_mass: Ratio for embedding LR (default: 32.0)
        num_layers: Number of transformer blocks (default: 3)
    
    Returns:
        Tuple of (muon_param_groups, embedding_params)
    """
    muon_param_groups = []
    embedding_params = []
    
    # Collect embedding parameters (handled separately)
    if hasattr(model, 'token_embedding'):
        embedding_params.append(model.token_embedding.weight)
    
    # Transformer block parameters with per-layer LR decay
    # From paper: "we decayed the learning rate by a factor of 1/2 per residual layer,
    # causing later layers to train more than earlier layers"
    decay_factor = 0.5
    for layer_idx in range(num_layers):
        layer_params = []
        for name, param in model.named_parameters():
            if f'blocks.{layer_idx}.' in name:
                layer_params.append(param)
        
        if layer_params:
            # Later layers get higher LR:
            # Layer 0: base_lr * 0.5^2 = 0.25 * base_lr
            # Layer 1: base_lr * 0.5^1 = 0.5 * base_lr
            # Layer 2: base_lr * 0.5^0 = 1.0 * base_lr
            layer_lr = base_lr * (decay_factor ** (num_layers - 1 - layer_idx))
            muon_param_groups.append({
                'params': layer_params,
                'lr': layer_lr,
                'name': f'blocks.{layer_idx}'
            })
    
    # Final layer norm and logit head (no LR decay, just base_lr)
    other_params = []
    for name, param in model.named_parameters():
        if 'token_embedding' not in name and 'blocks.' not in name:
            other_params.append(param)
    
    if other_params:
        muon_param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'other'
        })
    
    return muon_param_groups, embedding_params


def train(model_type: str = 'lipschitz_bounded'):
    """
    Train a Lipschitz-constrained transformer model.
    
    Args:
        model_type: Type of model to train. Options:
            - 'lipschitz_bounded': LipschitzBoundedTransformer
            - 'l2_attention': L2Transformer
    """
    # Select training config based on model type
    if model_type == 'lipschitz_bounded':
        TRAINING_CONFIG = LIPSCHITZ_BOUNDED_TRAINING_CONFIG
    elif model_type == 'l2_attention':
        TRAINING_CONFIG = L2_TRAINING_CONFIG
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Hyperparameters from paper Appendix F
    context_length = TRAINING_CONFIG['context_length']  # 256
    batch_size = TRAINING_CONFIG['batch_size']          # 64
    num_steps = TRAINING_CONFIG['num_steps']            # 2000
    eval_interval = 100
    
    # Muon hyperparameters
    base_lr = 0.02  # Muon uses higher LR than AdamW
    blocks_mass = 32.0  # From paper: embeddings learn at base_lr / 32
    num_layers = 3
    sigma_max = 1.0  # Spectral constraint
    max_inflation = 16.0  # Max inflation for embedding column normalization
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading Shakespeare dataset...")
    data = load_shakespeare(context_length=context_length, batch_size=batch_size)
    train_loader = data['train_loader']
    val_loader = data['test_loader']
    vocab_size = data['vocab_size']
    decode = data['decode']
    
    print(f"Vocab size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model using unified factory
    print(f"Creating {model_type} model...")
    model = create_model(model_type=model_type, vocab_size=vocab_size)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create parameter groups
    muon_param_groups, embedding_params = get_parameter_groups(
        model, base_lr, blocks_mass, num_layers
    )
    
    print("\nParameter groups:")
    for group in muon_param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(f"  Muon - {group['name']}: lr={group['lr']:.6f}, params={n_params:,}")
    if embedding_params:
        n_embed = sum(p.numel() for p in embedding_params)
        print(f"  AdamW - embeddings: lr={base_lr / blocks_mass:.6f}, params={n_embed:,}")
    
    # Optimizers
    # Muon for linear weights (with per-layer LR decay)
    optimizer_muon = Muon(muon_param_groups, lr=base_lr, momentum=0.95)
    
    # AdamW for embeddings (lower LR due to blocks_mass)
    optimizer_embed = torch.optim.AdamW(
        embedding_params, 
        lr=base_lr / blocks_mass,
        weight_decay=0.0
    )
    
    # Spectral constraint
    spectral_constraint = SpectralConstraint(model, sigma_max=sigma_max)
    
    # Training loop
    print(f"\nStarting training for {num_steps} steps with Muon optimizer...")
    print(f"  Base LR: {base_lr}, Embedding LR: {base_lr / blocks_mass:.6f}")
    print(f"  Spectral constraint: sigma_max={sigma_max}")
    model.train()
    
    step = 0
    train_iter = iter(train_loader)
    
    progress_bar = tqdm(total=num_steps, desc="Training")
    
    while step < num_steps:
        # Get batch (cycle through data if needed)
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Create causal mask
        seq_len = inputs.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        # Forward pass
        logits = model(inputs, mask=causal_mask)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        
        # Backward pass
        optimizer_muon.zero_grad()
        optimizer_embed.zero_grad()
        loss.backward()
        
        # Normalize embedding gradients (column normalization with max inflation)
        # From paper: "normalizing the columns of embedding gradient"
        if hasattr(model, 'token_embedding') and model.token_embedding.weight.grad is not None:
            model.token_embedding.weight.grad = normalize_embedding_columns(
                model.token_embedding.weight.grad,
                max_inflation=max_inflation
            )
        
        # Optimizer steps
        optimizer_muon.step()
        optimizer_embed.step()
        
        # Apply spectral constraint after optimizer steps
        spectral_constraint.step()
        
        # Normalize embedding weights to maintain ||x_0||_{∞RMS} ≤ 1
        # From paper: "Every step, RMS normalize the embedding columns."
        if hasattr(model, 'normalize_embeddings'):
            model.normalize_embeddings()
        
        # Update progress
        progress_bar.update(1)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluate periodically
        if (step + 1) % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, vocab_size)
            print(f"\nStep {step + 1}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
            
            # Generate sample
            sample = generate(model, device, vocab_size, decode, max_tokens=100)
            print(f"Sample: {sample[:200]}...")
            
            model.train()
        
        step += 1
    
    progress_bar.close()
    print("\nTraining complete!")
    
    # Final evaluation
    val_loss = evaluate(model, val_loader, device, vocab_size)
    print(f"Final validation loss: {val_loss:.4f}")

    # Lipschitz constraints:
    result = compute_lipschitz_certificate(model)
    print_lipschitz_report(result)
    
    # Save model
    model_filename = f'model_{model_type}.pt'
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")
    
    return model


@torch.no_grad()
def evaluate(model, val_loader, device, vocab_size):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        seq_len = inputs.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        logits = model(inputs, mask=causal_mask)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        
        total_loss += loss.item()
        num_batches += 1
        
        # Only evaluate on a subset for speed
        if num_batches >= 10:
            break
    
    return total_loss / num_batches


@torch.no_grad()
def generate(model, device, vocab_size, decode, max_tokens=100, temperature=1.0):
    """Generate text from the model."""
    model.eval()
    
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    for _ in range(max_tokens):
        idx_cond = idx[:, -256:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return decode(idx[0].tolist())


if __name__ == "__main__":
    train("lipschitz_bounded")
