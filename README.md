# Comparing Lipschitz Transformer models

This repository is part of an essay on "Lipschitz-Constrained Transformers". I implemented different ways of bounding the Lipschitz constant of Transformer models.

### Background:
A function $f: \mathbb{R}^{n} \to \mathbb{R}^n$ is called Lipschitz-continuous with Lipschitz constant $L$ if:
$$\|f(x)-f(y)\| \le L \|x-y\|$$
for some norm $\|\cdot \|$ on $\mathbb{R}^{n}$ (e.g. the L2 Norm).

We can see a Transformer as a function $f: \mathbb{R}^{N \times D} \to \mathbb{R}^{N \times D}$, where $N$ is the sequence length and $D$ the dimension of the model (e.g. the dimension of the embedding vectors). In my essay, I am looking at different ways to make the Transformer Lipschitz-continuous and constrain the Lipschitz continous.

There are several motivations to look at Lipschitz-continuity in Transformers, e.g.:
- Lipschitz continuity gives a provable guarantee of adversarial robustness in Computer Vision Applications
- The Lipschitz constant bounds the gradient preventing exploding gradients and stabilizing training. It is possible to train Transformers with enforced Lipschitz constraints with out Learning Rate warmup and LayerNorm.

### Goal of this repository: 
I implement and compare several different methods of constraining Transformer models and test them in a NLP application to compare their performance vs. their Lipschitz constant.

As the normal Dot-Product Attention is not globally Lipschitz, there are two differents ways to achieve Lipschitz constraints. Either we bound the activations of every layer to use that the attention mechanism is continuous differentiable to get a Lipschitz bound for the attention layer or we change the attention Layer to be globally Lipschitz by replacing it by L2-Attention or Scaled-Dot-Product Attention.
In this code, I implement one way of doing the prior (see ./models/dot_product_attention.py) and two ways of doing the later (see ./models/l2_attention_model.py and ./models/lipsformer.py)
