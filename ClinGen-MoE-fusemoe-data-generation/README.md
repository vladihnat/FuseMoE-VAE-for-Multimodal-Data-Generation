# Multimodal Synthetic Data Generation with MoE (Minimum Viable Product)

This repository contains the **first implementation (MVP)** of a multimodal generative architecture designed to explore **synthetic data generation using Mixture-of-Experts (MoE)** mechanisms.

The project is part of the **TER: *Mixtures of Experts – Adapted Foundation Models for the Synthetic Generation of Multimodal Clinical Data***.

The goal is to explore **frugal multimodal generative architectures** capable of learning shared latent representations across heterogeneous modalities and generating synthetic samples.

## LLM Context

This repository includes `LLM_CONTEXT.md`, a file designed to help coding agents
understand the architecture of the project.

If using an AI coding assistant:

1. Provide `LLM_CONTEXT.md` as repository context
2. Ask the agent to read it before answering questions

This ensures the agent respects the modular architecture. A typical prompt to set up would look like : 

>Before answering any question:
>1. Read LLM_CONTEXT.md completely.
>2. Use it as the architecture specification for the project.
>
>Respect the modular structure of:
>encoders -> fusion -> latent -> decoders -> losses -> training.
>
>When proposing changes:
>- avoid breaking module interfaces
>- keep implementations local to their modules
>- follow the extension rules described in LLM_CONTEXT.md.

---

# Current Status (MVP)

This repository implements the **first functional version of the architecture**, including:

- ✅ **Encoders**
- ✅ **Fusion block (MoE-based fusion)**
- ✅ **Latent representation module**
- ✅ **Tabular decoder**
- ❗ **Time-Series decoder (TODO)**

The **Time-Series (TS) decoder is intentionally left unimplemented for now** because its design requires careful consideration regarding:

- temporal reconstruction strategy
- sequence length handling
- irregular time sampling
- decoder architecture (RNN, Transformer, diffusion-style, etc.)

These design choices are **not trivial**, and implementing a naive solution could bias the architecture prematurely. Therefore, the TS decoder will be implemented **after validating the core generative pipeline**.

---

# Data Strategy (Important)

At this stage we **do NOT use real clinical datasets**.

Instead, we rely on **synthetic data** for the following reasons:

- avoid the complexity of clinical data preprocessing
- isolate and debug architectural components
- validate the **generative pipeline independently of dataset constraints**
- accelerate development of the MVP

Real datasets (e.g. MIMIC-III / MIMIC-IV) will be integrated **only after the architecture is validated**.

---

# Architecture

The generative pipeline follows the modular structure:

> Encoders → Fusion → Latent → Decoders → Reconstructors


### 1. Encoders
Each modality is processed by a **dedicated encoder** that maps raw input data into a shared latent embedding space.

Examples:
- tabular encoder
- time-series encoder

These encoders transform heterogeneous inputs into **compatible representations**.

---

### 2. Fusion Module (MoE-based)

The encoded modality embeddings are fused using a **Mixture-of-Experts inspired fusion block**.

This module learns to combine modality-specific information into a **shared multimodal representation**.

Key idea:
- each modality acts as a specialized expert
- the fusion mechanism aggregates the information into a joint representation

---

### 3. Latent Representation

The fused representation is then projected into a **latent space**.

This latent variable is intended to capture the **joint multimodal distribution** of the data.

In future training phases, this block will be optimized using **variational objectives (e.g., VAE-style ELBO)**.

---

### 4. Decoders

Each modality is reconstructed through a **modality-specific decoder**.

Current status:

| Modality | Decoder Status |
|--------|--------|
| Tabular | ✅ Implemented |
| Time-series | ❗ TODO |

The tabular decoder reconstructs:
- continuous variables
- categorical variables (via multiple categorical heads)

---

### 5. Reconstruction

The reconstructed outputs are used to compute:

- reconstruction losses
- regularization losses (e.g., KL divergence in VAE setting)
- possible MoE balancing losses

These components will be used during **training of the full generative model**.

---

# Repository Structure 


```
fusemoe_gen/
├─ README.md
├─ requirements.txt
├─ LLM_CONTEXT.md
├─ data/
│  ├─ generate_synthetic_ts_tab.py
│  ├─ raw/
│  └─ processed/
│     └─ synthetic_ts_tab/
|        ├─ metada.json
|        ├─ test.pkl
|        ├─ train.pkl 
|        └─ val.pkl 
├─ src/
│  ├─ __init__.py
│  ├─ data/
│  │  └─ datasets.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ multimodal_vae.py
│  │  ├─ encoders/
│  │  │  ├─ __init__.py
│  │  │  ├─ ts_irregular.py
│  │  │  └─ tabular.py
│  │  ├─ fusion/
│  │  │  ├─ __init__.py
│  │  │  ├─ sparse_moe.py
│  │  │  └─ router_utils.py
│  │  ├─ latent/
│  │  │  ├─ __init__.py
│  │  │  └─ posterior.py
│  │  ├─ decoders/
│  │  │  ├─ __init__.py
│  │  │  └─ tabular_decoder.py
│  ├─ losses/
│  │  ├─ __init__.py
│  │  ├─ reconstruction.py
│  │  ├─ total.py
│  │  └─ kl.py
│  └─ training/
│     ├─ __init__.py
│     └─ engine.py
└─ tests/
   ├─ __init__.py
   ├─ conftest.py
   ├─ pytests/
   |  ├─ __init__.py
   |  ├─ test_datasets.py
   |  ├─ test_sparse_moe_fusion.py
   |  ├─ test_ts_encoder.py
   |  ├─ test_tabular_reconstruction.py
   |  ├─ test_tabular_encoder.py
   |  ├─ test_tabular_decoder.py
   |  └─ test_training_engine.py 
   └─ smokes/
      ├─ __init__.py
      ├─ smokeFusion.py
      ├─ smokePosterior.py
      ├─ smokeSyntData.py
      └─ smokeTabDecoder.py 
```
