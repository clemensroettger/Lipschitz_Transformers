"""Models package with unified factory function."""

from .dot_product_attention_model import (
    LipschitzBoundedTransformer,
    create_model as create_lipschitz_bounded_model,
    MODEL_CONFIG as LIPSCHITZ_BOUNDED_CONFIG,
    TRAINING_CONFIG as LIPSCHITZ_BOUNDED_TRAINING_CONFIG,
    print_lipschitz_report,
    compute_lipschitz_certificate
)

from .l2_attention_model import (
    L2Transformer,
    create_model as create_l2_model,
    MODEL_CONFIG as L2_CONFIG,
    TRAINING_CONFIG as L2_TRAINING_CONFIG,
)


def create_model(model_type: str, vocab_size: int, **kwargs):
    """
    Unified factory function to create any Lipschitz-constrained transformer model.
    
    Args:
        model_type: Type of model to create. Options:
            - 'lipschitz_bounded': LipschitzBoundedTransformer
            - 'l2_attention': L2Transformer
        vocab_size: Vocabulary size
        **kwargs: Model-specific configuration overrides
    
    Returns:
        Transformer model instance
    
    Example:
        >>> model = create_model('lipschitz_bounded', vocab_size=65, d_model=256)
        >>> model = create_model('l2_attention', vocab_size=65, num_layers=3)
    """
    if model_type == 'lipschitz_bounded':
        return create_lipschitz_bounded_model(vocab_size, **kwargs)
    elif model_type == 'l2_attention':
        return create_l2_model(vocab_size, **kwargs)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available options: 'lipschitz_bounded', 'l2_attention'"
        )


# Export all models and configs
__all__ = [
    # Models
    'LipschitzBoundedTransformer',
    'L2Transformer',
    # Factory function
    'create_model',
    # Individual factory functions (for direct access if needed)
    'create_lipschitz_bounded_model',
    'create_l2_model',
    # Configs
    'LIPSCHITZ_BOUNDED_CONFIG',
    'LIPSCHITZ_BOUNDED_TRAINING_CONFIG',
    'L2_CONFIG',
    'L2_TRAINING_CONFIG',
]
