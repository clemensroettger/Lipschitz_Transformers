"""
Training script for LipschitzBoundedTransformer on Shakespeare dataset.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from load_shakespeare import load_shakespeare
from models.model_lipschitz_bounds import create_model, TRAINING_CONFIG
from utils import SpectralConstraint


def train():
    # Hyperparameters
    context_length = TRAINING_CONFIG['context_length']  # 256
    batch_size = TRAINING_CONFIG['batch_size']          # 64
    learning_rate = TRAINING_CONFIG['learning_rate']    # 1e-3
    num_steps = TRAINING_CONFIG['num_steps']            # 2000
    eval_interval = 100  # Evaluate every N steps
    
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
    
    # Create model
    print("Creating model...")
    model = create_model(vocab_size=vocab_size)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Spectral constraint
    spectral_constraint = SpectralConstraint(model)
    spectral_constraint.step()
    
    # Training loop
    print(f"\nStarting training for {num_steps} steps...")
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
        
        # Create causal mask (lower triangular) to prevent seeing future tokens
        seq_len = inputs.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        # Forward pass with causal mask
        logits = model(inputs, mask=causal_mask)  # [batch, seq_len, vocab_size]
        
        # Compute loss (cross-entropy)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        spectral_constraint.step()
        
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
    
    # Save model
    torch.save(model.state_dict(), 'model_lipschitz_bounds.pt')
    print("Model saved to model_lipschitz_bounds.pt")
    
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
        
        # Create causal mask for validation (same as training)
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
    
    # Start with a random token
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    for _ in range(max_tokens):
        # Crop to max sequence length
        idx_cond = idx[:, -256:]
        
        # Get predictions
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # Last token only
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    # Decode and return
    return decode(idx[0].tolist())


if __name__ == "__main__":
    train()
