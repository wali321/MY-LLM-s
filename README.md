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

The repository documents three distinct releases, preserving each version's implementation and evidence scope independently:

### 1. Vectora v1.0 — Architecture Baseline
* **Goal**: Understand Transformer architecture, tensor shapes, and forward-pass data flow.
* **Tokenization**: Subword byte-level BPE via `tiktoken` (GPT-2 encoding, 50,257 vocabulary).
* **Architecture**: 4 Transformer blocks, 4 attention heads, 256 embedding dimension, 128 context length.
* **Weights**: Randomly initialized (28.9M parameters, untied embeddings).
* **Decoding**: Greedy autoregressive next-token prediction.
* **Role**: Serves as the structural baseline for decoder-only architectures without training.
* 📄 [Read Vectora v1.0 Documentation](https://vectoradocs.vercel.app/models/vectora/v1)

### 2. Vectora v1.1 — Trained Character-Level Model
* **Goal**: Implement an end-to-end training pipeline and learn next-character prediction on a real corpus.
* **Tokenization**: Character-level vocabulary (83 unique characters extracted from the corpus subset).
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

### 3. Vectora v1.2 — 23M Dataset Experiment (Latest)
* **Goal**: Scale training to the entire 82M-character corpus (~23.4M estimated GPT-2 tokens) over 50,000 iterations to evaluate scaling dynamics, long-horizon loss convergence, and character representation stability on an expanded character set.
* **Tokenization**: Custom character-level vocabulary consisting of 333 unique characters extracted across the full dataset.
  > **Technical Accuracy Note**: The ~23.4M figure represents an estimated GPT-2 subword tokenizer token count used as a comparative scale benchmark. The actual model training strictly employs character-level tokenization over the 333-character vocabulary.
* **Dataset Statistics**:
  * **Total characters**: 82,215,804
  * **Total lines**: 2,803,655
  * **GPT-2 estimated tokens**: 23,395,305
  * **Training split**: 73,994,223 characters (90%)
  * **Validation split**: 8,221,581 characters (10%)
  * **Vocabulary size**: 333 characters
* **Architecture Specifications**:
  * **Model Type**: GPT-style decoder-only Transformer
  * **Total Parameters**: 695,168
  * **Transformer Blocks**: 3
  * **Attention Heads**: 4 ($d_{\text{head}} = 32$)
  * **Embedding Dimension**: 128
  * **Context Window ($T$)**: 128 characters
  * **Feed-Forward Dimension**: 512 ($4 \times d_{\text{model}}$)
  * **Dropout**: 0.1
  * **Weight Tying**: Tied input/output embeddings (`lm_head.weight = token_embedding.weight`)
  * **Attention**: Causal self-attention with lower-triangular mask
  * **Activation**: Exact `nn.GELU()`
  * **Normalization**: Pre-LayerNorm residual connections
* **Training Configuration**:
  * **Framework**: PyTorch
  * **Backend**: DirectML (with automatic CUDA / CPU fallback)
  * **Batch Size**: 16
  * **Training Iterations**: 50,000
  * **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $= 0.1$)
  * **Initial Learning Rate**: 0.0003 ($3 \times 10^{-4}$)
  * **Learning Rate Schedule**: Linear warmup (50 steps) + Cosine decay to $3 \times 10^{-5}$
  * **Gradient Clipping**: 1.0 max norm
* **Training Results**:
  * **Final Logged Training Loss**: **1.3793**
  * **Final Logged Validation Loss**: **1.4082**
  * **Status**: 50,000 iterations successfully completed
* **Decoding**: Temperature-scaled top-$k$ sampling ($T=0.5, k=40$).
* 📄 [Read Vectora v1.2 Documentation](https://vectoradocs.vercel.app/models/vectora/v1.2)

---

## Technical Comparison

| Feature | Vectora v1.0 | Vectora v1.1 | Vectora v1.2 (23M Experiment) |
|---|---|---|---|
| **Architecture** | Decoder-only Transformer | Decoder-only Transformer | Decoder-only Transformer |
| **Tokenization** | Subword BPE (`tiktoken` GPT-2) | Character-level (`sorted(set(text))`) | Character-level (`sorted(set(text))`) |
| **Vocabulary Size** | 50,257 | 83 | 333 |
| **Context Length ($T$)** | 128 tokens | 128 characters | 128 characters |
| **Embedding Dim ($d_{\text{model}}$)** | 256 | 128 | 128 |
| **Transformer Layers ($L$)** | 4 | 3 | 3 |
| **Attention Heads ($h$)** | 4 ($d_{\text{head}} = 64$) | 4 ($d_{\text{head}} = 32$) | 4 ($d_{\text{head}} = 32$) |
| **MLP Width** | 1,024 ($4 \times d$) | 512 ($4 \times d$) | 512 ($4 \times d$) |
| **Total Parameters** | 28,920,832 | 631,168 | 695,168 |
| **Weight Tying** | Untied | Tied (`lm_head.weight = token_embedding.weight`) | Tied (`lm_head.weight = token_embedding.weight`) |
| **Q/K/V Projection** | Separate linear layers (no bias) | Fused linear layer (no bias) | Fused linear layer (no bias) |
| **Activation** | Tanh-approximated GELU | Exact `nn.GELU()` | Exact `nn.GELU()` |
| **Trained Status** | Untrained (random init baseline) | Fully trained (10,000 steps) | Fully trained (50,000 steps) |
| **Training Dataset** | None (forward demo only) | Shakespeare text (500k char subset) | Expanded corpus (73.99M train chars / 82.22M total) |
| **Corpus Scale Benchmark**| N/A | ~500k chars | ~23,395,305 estimated GPT-2 tokens (82.2M chars) |
| **Optimizer** | None | AdamW ($\beta_1=0.9, \beta_2=0.95$, decay=0.1) | AdamW ($\beta_1=0.9, \beta_2=0.95$, decay=0.1) |
| **LR Schedule** | None | 50-step warmup + Cosine decay | 50-step warmup + Cosine decay |
| **Training Steps** | None | 10,000 | 50,000 |
| **Final Loss (Train / Val)** | N/A | **1.3449 / 1.4849** | **1.3793 / 1.4082** |
| **Sampling Method** | Greedy argmax | Temperature-scaled top-$k$ sampling | Temperature-scaled top-$k$ sampling |
| **Checkpoint** | None | `fcc.pt` (CPU portable dictionary) | `fcc.pt` (CPU portable dictionary) |

---

## What Changed in v1.2

1. **Substantial Dataset Scaling**:
   * v1.1 trained exclusively on the first 500,000 characters of the corpus.
   * v1.2 trains on **73,994,223 characters** (from the full 82,215,804 character corpus, with 8,221,581 characters reserved for validation) across 2,803,655 lines, providing roughly 23.4 million estimated GPT-2 equivalent subword tokens of text exposure.
2. **Vocabulary Expansion**:
   * Expanded character set from **83** to **333** unique characters, supporting a richer set of symbols, special punctuation, and unicode formatting without truncation.
3. **Parameter Increase via Embedding Dimensions**:
   * Increased parameter count from **631,168** to **695,168** (+64,000 effective projection elements, or 32,000 unique parameters due to tied input and output embeddings: $333 \times 128$ vs $83 \times 128$).
4. **Extended Training Horizon**:
   * Training steps increased by 5× from **10,000** to **50,000 iterations**, processing over 102.4 million tokens in total ($50,000 \times 16 \times 128$).
5. **Lower Validation Loss and Improved Generalization**:
   * Validation loss reached **1.4082** (down from 1.4849 in v1.1) despite evaluating on a validation split that is over 160× larger (8.22M characters vs 50k characters).

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
│   ├── train.txt                      # Shakespeare corpus subset (~86MB full source)
│   └── fcc.pt                         # Saved model weights checkpoint
│
├── Vectora-v1.2-23M/                  # Vectora v1.2 23M dataset experiment
│   ├── V1.2-23M.ipynb                 # 50,000-iteration training loop on full 82M char dataset
│   └── train.txt                      # Full training corpus (82,215,804 chars, 2,803,655 lines)
│
└── README.md                          # Repository overview and documentation
```

---

## Sample Outputs

### Vectora v1.2 Output

Generated after 50,000 steps using prompt `"bade the voyagers"` ($T=0.5, k=40, \text{max\_tokens}=300$):

```text
bade the voyagers after a short of his long

for her life and she left of the providence, and the school recognized

the part of the soul short of the south.

“The same second man at the state,” said Dorothea, “but a serious in the

same of the servant of the evening body of the priest proceeding the

way in the cha
```

*Observation*: With the expanded corpus and 50,000 iterations, the model forms coherent narrative prose structures, conversational dialogue with quotation marks, and standard multi-clause syntax while operating purely on character predictions.

### Vectora v1.1 Output

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
git clone https://github.com/wali321/MY-LLM-s.git
cd MY-LLM-s

# Install dependencies
pip install torch torchvision torchaudio tiktoken jupyter
```

### Running Vectora v1.0

```bash
cd Vectora-v1.0
jupyter notebook v1.ipynb
```

### Running Vectora v1.1

```bash
cd Vectora-v1.1
jupyter notebook v1.1.ipynb
```

### Running Vectora v1.2 (23M Dataset Experiment)

```bash
cd Vectora-v1.2-23M
jupyter notebook V1.2-23M.ipynb
```

The notebook automatically detects your best available compute backend (`DirectML`, `CUDA`, or `CPU`), encodes the complete 82.2M-character `train.txt` corpus into character token IDs, executes the 50,000-step training loop with warmup and cosine decay, logs validation loss, saves the checkpoint `fcc.pt`, and performs autoregressive sampling.

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
