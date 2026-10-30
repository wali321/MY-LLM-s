# Vectora

[![Live Documentation](https://img.shields.io/badge/Docs-vectoradocs.vercel.app-blue?style=flat-square)](https://vectoradocs.vercel.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DirectML%20%7C%20CUDA%20%7C%20CPU-EE4C2C?style=flat-square)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

An educational machine learning project exploring the internal mechanics of decoder-only Transformers by building and training them from scratch in Python and PyTorch.

Instead of treating large language models as black-box APIs, **Vectora** implements every component directly: character and subword tokenization, token and positional embeddings, multi-head causal self-attention, feed-forward networks with GELU activations, pre-LayerNorm residual connections, training optimization with learning rate schedules, and autoregressive generation.

📖 **Official Documentation**: [https://vectoradocs.vercel.app](https://vectoradocs.vercel.app)  
📊 **Models Directory**: [https://vectoradocs.vercel.app/models](https://vectoradocs.vercel.app/models)

---

## Model Family

The repository documents two distinct releases, preserving each version's implementation and evidence scope independently:

### 1. Vectora v1.0 — Architecture Baseline
* **Goal**: Understand Transformer architecture, tensor shapes, and forward-pass data flow.
* **Tokenization**: Subword byte-level BPE via `tiktoken` (GPT-2 encoding, 50,257 vocabulary).
* **Architecture**: 4 Transformer blocks, 4 attention heads, 256 embedding dimension, 128 context length.
* **Weights**: Randomly initialized (28.9M parameters, untied embeddings).
* **Decoding**: Greedy autoregressive next-token prediction.
* **Role**: Serves as the structural baseline for decoder-only architectures without training.
* 📄 [Read Vectora v1.0 Documentation](https://vectoradocs.vercel.app/models/vectora/v1)

### 2. Vectora v1.1 — Trained Character-Level Model (Latest)
* **Goal**: Implement an end-to-end training pipeline and learn next-character prediction on a real corpus.
* **Tokenization**: Character-level vocabulary (83 unique characters extracted from the corpus).
* **Dataset**: Shakespeare text corpus (trained on the first 500,000 characters; 90/10 train/validation split).
* **Architecture**: 3 Transformer blocks, 4 attention heads, 128 embedding dimension, 128 context length (631,168 parameters).
* **Optimizations**: Tied input/output embeddings, exact GELU, no embedding dropout.
* **Training Pipeline**:
  * **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $= 0.1$)
  * **Schedule**: Linear warmup (50 steps) + Cosine decay to $\text{lr}_{\min} = 3 \times 10^{-5}$
  * **Peak LR**: $3 \times 10^{-4}$
  * **Gradient Clipping**: $1.0$ max norm
  * **Iterations**: 10,000 steps ($B=16, T=128$, 2,048 tokens/step)
* **Results**:
  * Training loss: $4.4233 \to 1.3449$
  * Validation loss: $4.4231 \to 1.4849$
* **Decoding**: Temperature-scaled top-$k$ sampling ($T=0.5, k=40$).
* 📄 [Read Vectora v1.1 Documentation](https://vectoradocs.vercel.app/models/vectora/v1.1)

---

## Technical Comparison

| Feature | Vectora v1.0 | Vectora v1.1 |
|---|---|---|
| **Architecture** | Decoder-only Transformer | Decoder-only Transformer |
| **Tokenization** | Subword BPE (`tiktoken` GPT-2) | Character-level (`sorted(set(text))`) |
| **Vocabulary Size** | 50,257 | 83 |
| **Context Length ($T$)** | 128 tokens | 128 characters |
| **Embedding Dim ($d_{\text{model}}$)** | 256 | 128 |
| **Transformer Layers ($L$)** | 4 | 3 |
| **Attention Heads ($h$)** | 4 ($d_{\text{head}} = 64$) | 4 ($d_{\text{head}} = 32$) |
| **MLP Width** | 1,024 ($4 \times d$) | 512 ($4 \times d$) |
| **Total Parameters** | 28,920,832 | 631,168 |
| **Weight Tying** | Untied | Tied (`lm_head.weight = token_embedding.weight`) |
| **Q/K/V Projection** | Separate linear layers (no bias) | Fused linear layer (no bias) |
| **Activation** | Tanh-approximated GELU | Exact `nn.GELU()` |
| **Trained Status** | Untrained (random init baseline) | Fully trained (10,000 steps) |
| **Training Dataset** | None (forward demo only) | Shakespeare text (500k char subset) |
| **Optimizer** | None | AdamW ($\beta_1=0.9, \beta_2=0.95$, decay=0.1) |
| **LR Schedule** | None | 50-step warmup + Cosine decay |
| **Final Loss (Train / Val)** | N/A | **1.3662 / 1.4849** (perplexity $\sim 4.4$) |
| **Sampling Method** | Greedy argmax | Temperature-scaled top-$k$ sampling |
| **Saved Checkpoint** | None | `fcc.pt` (CPU portable dictionary) |

---

## Repository Structure

```
Vectora/
├── Vectora-v1.0/                      # Vectora v1.0 architecture implementation
│   ├── v1.ipynb                       # PyTorch implementation notebook (GPT-2 tokenization)
│   └── wizardofoz.txt                 # Text slice for demonstration forward pass
│
├── Vectora-v1.1/                      # Vectora v1.1 trained character-level model
│   ├── v1.1.ipynb                     # Complete training loop, validation, and generation
│   ├── train.txt                      # Shakespeare corpus (~86MB)
│   └── fcc.pt                         # Saved model weights checkpoint
│
└── README.md                          # Repository overview and documentation
```

---

## Sample Output (Vectora v1.1)

Generated after 10,000 steps using prompt `"bade the voyagers"` ($T=0.5, k=40, \text{max\_tokens}=300$):

```text
bade the voyagers of the brother than the world,

The time of my be subject stand of the world,

That shall thou shalt that honour she dispered,

Where thou art a conquertant his entreath,

And the live the content of the leaves still,

And beauty the honour that that have in my life,

Thou hast thy rance disted the
```

The model successfully learns English orthography, Shakespearean vocabulary (*thou*, *hast*, *thy*, *shalt*, *art*), and approximate poetic meter from character sequences alone.

---

## Getting Started

### Prerequisites

* Python 3.10+
* PyTorch 2.0+
* Jupyter Notebook / JupyterLab

Optional hardware acceleration:
* **DirectML** on Windows: `pip install torch-directml` (supports AMD Radeon & Intel GPUs)
* **CUDA**: NVIDIA GPU with compatible CUDA toolkit
* **CPU**: Works out of the box on standard CPUs

### Installation

```bash
# Clone the repository
git clone https://github.com/wali321/Vectora.git
cd Vectora

# Install dependencies
pip install torch torchvision torchaudio tiktoken jupyter
```

### Running Vectora v1.0

```bash
cd Vectora-v1.0
jupyter notebook v1.ipynb
```

### Running Vectora v1.1 (Training & Generation)

```bash
cd Vectora-v1.1
jupyter notebook v1.1.ipynb
```

The notebook will automatically detect your best available compute backend (`DirectML`, `CUDA`, or `CPU`), preprocess `train.txt`, build the vocabulary, execute the 10,000-step training loop, print validation loss every 50 steps, save `fcc.pt`, and generate text from a prompt.

---

## Hardware and Acceleration

Vectora notebooks are written with automatic multi-backend support:

```python
try:
    import torch_directml
    device = torch_directml.device()
    BACKEND = "DirectML"
except (ImportError, RuntimeError, Exception):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        BACKEND = "CUDA"
    else:
        device = torch.device("cpu")
        BACKEND = "CPU"
```

This allows seamless execution on Windows machines equipped with AMD Radeon graphics via DirectML, NVIDIA machines via CUDA, or standard CPU environments.

---

## References & Learning Resources

* **Vaswani et al. (2017)**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* **Radford et al. (2019)**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (GPT-2)
* **Sebastian Raschka**: [Build a Large Language Model (From Scratch)](https://sebastianraschka.com/books.html)
* **Andrej Karpathy**: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* **Peeyush K Misra**: Building LLM From Scratch series

---

## Author

**Syed Wali Haider**  
GitHub: [@wali321](https://github.com/wali321)  
Project Documentation: [https://vectoradocs.vercel.app](https://vectoradocs.vercel.app)
