# LLM Context for `fusemoe_gen`

## Purpose

This file is meant to help an LLM agent understand, navigate, extend, and debug the current MVP codebase after cloning the repository.

The repository implements a **modular multimodal generative prototype** built around:

- modality-specific encoders,
- sparse Mixture-of-Experts fusion,
- a latent posterior head,
- modality decoders,
- reconstruction and KL losses,
- and a small training engine.

The current MVP is mainly focused on **two modalities**:

- **tabular**
- **irregular time-series**

The codebase is designed so that new modalities, fusion variants, latent heads, and decoders can be added with minimal changes outside their own module.

---

## High-level pipeline

```text
inputs
  -> encoders
  -> fusion
  -> latent posterior
  -> decoders
  -> losses
  -> training
```

For the current MVP, the forward path is conceptually:

```text
time-series + tabular
  -> modality encoders
  -> sparse MoE fusion
  -> posterior head
  -> latent sample z
  -> decoder(s)
  -> reconstruction loss + KL loss
```

This is a research MVP, not yet a full production-ready generative framework.

---

## Repository structure


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


---

## Module responsibilities

### `src/data/datasets.py`
Owns dataset loading and sample formatting.

Typical responsibilities:
- load serialized synthetic splits,
- expose samples in a PyTorch-friendly format,
- centralize shape conventions for tabular and time-series inputs.

When extending the project, prefer updating dataset logic here instead of scattering preprocessing assumptions across the model code.

---

### `src/models/encoders/`

Contains **one encoder per modality**.

#### `tabular.py`
Encodes fixed-size tabular features into a learned embedding.

Typical contract:
- input: `[batch, tab_dim]`
- output: `[batch, d_model]` or similar

#### `ts_irregular.py`
Encodes irregular time-series inputs into a fixed-size representation usable by fusion.

This is the place to evolve:
- time handling,
- masks,
- variable-length sequences,
- pooling,
- temporal attention / RNN / transformer variants.

**Extension rule:** new modality = new encoder file with a small interface returning an embedding for fusion.

---

### `src/models/fusion/`

Contains multimodal fusion logic.

#### `sparse_moe.py`
Core sparse Mixture-of-Experts fusion block.

Expected role:
- receive modality embeddings,
- compute router / gating scores,
- dispatch to experts,
- combine expert outputs into one fused representation.

#### `router_utils.py`
Helper utilities for routing and gating.

Keep things here such as:
- top-k helpers,
- score normalization,
- dispatch utilities,
- optional load-balancing helpers.

**Extension rule:** keep routing math/utilities here instead of overloading `sparse_moe.py`.

---

### `src/models/latent/`

Contains latent-space parameterization.

#### `posterior.py`
Maps fused features to posterior parameters.

Typical VAE-style role:
- input: fused embedding,
- output: latent parameters such as `mu` and `logvar`,
- optionally reparameterization helpers.

This module should stay independent from modality-specific code.

---

### `src/models/decoders/`

Contains modality-specific decoders.

#### `tabular_decoder.py`
Current implemented decoder for reconstructing tabular features from latent code.

Typical contract:
- input: latent vector `z`
- output: tabular reconstruction or decoder parameters

**Important MVP limitation:** time-series decoding is not yet the fully stabilized part of the current implementation, so contributors should treat it as an extension point rather than a finished block.

---

### `src/models/multimodal_vae.py`

Top-level model assembly.

This file should orchestrate:
- modality encoder calls,
- fusion,
- posterior parameterization,
- latent sampling,
- decoder calls,
- return values needed by the losses and training engine.

**Design rule:** global model logic goes here, but the details of each block stay in submodules.

---

### `src/losses/`

#### `reconstruction.py`
Owns reconstruction objectives.

Use this for:
- tabular reconstruction,
- future time-series reconstruction,
- modality-wise reconstruction aggregation.

#### `kl.py`
Owns KL divergence computation for the latent posterior.

Keep KL logic isolated so posterior changes do not leak into decoder code.

---

### `src/training/engine.py`

Minimal orchestration layer for training.

Typical responsibilities:
- unpack batch,
- call model forward,
- compute losses,
- aggregate the objective,
- run backward / optimizer steps,
- return metrics.

This file should remain thin and should not become the place where architecture logic lives.

---

## Data assumptions

The current MVP uses synthetic processed data stored in:

```text
data/processed/synthetic_ts_tab/
```

Expected files:
- `train.pkl`
- `val.pkl`
- `test.pkl`
- `metadata.json`

Generation utility:
```text
data/generate_synthetic_ts_tab.py
```

This means the current prototype is built first to validate shapes, plumbing, and training behavior on **synthetic tabular + time-series data**.

---

## Coding philosophy

### 1. One modality = one encoder / one decoder when possible
Do not hard-code modality logic into the global model.

### 2. Fusion should stay modality-agnostic
Fusion should work on embeddings, not on raw data-specific assumptions.

### 3. Latent logic should stay separate
The posterior head should not care whether the fused signal came from 2 or 5 modalities.

### 4. The training engine should orchestrate, not define architecture
Model math belongs in `src/models/`, losses in `src/losses/`.

### 5. Tests should validate interfaces and shapes first
For this MVP, interface consistency matters as much as raw performance.

---

## Common extension patterns

### Add a new modality
Minimal path:
1. add a new encoder in `src/models/encoders/`
2. optionally add a decoder in `src/models/decoders/`
3. update `multimodal_vae.py`
4. update `src/data/datasets.py`
5. update reconstruction loss if needed
6. add smoke test + unit test

Goal: extension should be additive, not require rewriting the current pipeline.

---

### Replace or improve fusion
Examples:
- dense fusion,
- cross-attention,
- hierarchical MoE,
- modality-specific experts.

Preferred path:
- keep the fusion interface stable,
- implement the variant in `src/models/fusion/`,
- keep routing helpers in `router_utils.py`.

---

### Replace posterior parameterization
Examples:
- richer posterior head,
- conditional prior,
- future normalizing-flow variant.

Preferred path:
- localize changes to `src/models/latent/` and `multimodal_vae.py`,
- preserve outputs expected by `kl.py` and `engine.py`.

---

### Add a time-series decoder
Likely next milestone.

Recommended path:
- create a dedicated decoder file in `src/models/decoders/`,
- define clearly whether it reconstructs full sequences, discretized sequences, or timestep-wise parameters,
- add its loss in `src/losses/reconstruction.py`,
- make the model return both tabular and time-series reconstructions explicitly.

Because this design choice is architectural, an agent should avoid making silent assumptions about:
- sequence length,
- interpolation,
- autoregressive vs non-autoregressive decoding.

---

## Smoke tests and tests

Smoke scripts:
- `smokeTest.py`
- `smokeFusion.py`
- `smokePosterior.py`
- `smokeTabDecoder.py`
- `smoke_train_step_.py`

Formal tests in `tests/` cover:
- dataset loading,
- encoders,
- sparse MoE fusion,
- decoder behavior,
- reconstruction,
- training integration.

### Debugging order for an agent
When something fails locally, inspect in this order:
1. import paths
2. tensor shapes
3. dataset field names
4. forward signatures
5. loss expectations
6. train-step return contract

In this repo, many bugs are likely to come from **module interface mismatch** rather than deep algorithmic errors.

---

## Likely shape contract

Exact sizes may evolve, but the reasoning contract is:

- tabular input -> tabular encoder -> tabular embedding
- irregular time-series input -> ts encoder -> ts embedding
- embeddings -> sparse MoE fusion -> fused embedding
- fused embedding -> posterior -> `mu`, `logvar`
- latent sample `z` -> decoder(s)
- decoder outputs + original targets -> reconstruction losses
- posterior params -> KL loss

Always verify:
- batch dimension consistency,
- embedding dimension consistency across encoders and fusion,
- latent dimension consistency between posterior and decoders,
- decoder output consistency with targets.

---

## Local setup assumptions

After cloning, contributors should generally be able to run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

And/or smoke tests such as:

```bash
python smokeFusion.py
python smokePosterior.py
python smokeTabDecoder.py
python smoke_train_step_.py
```

If import issues occur, the most likely cause is that commands are being run from the wrong working directory.

Recommended habit:
- run from the repository root,
- keep imports consistent with the `src/` structure,
- avoid renaming package paths unless the whole repo is updated coherently.

---

## Known MVP boundaries

An agent should keep these assumptions explicit:
- this is an MVP, not a finished multimodal generator,
- the emphasis is on code structure and first working blocks,
- tabular decoding is more concrete than time-series decoding,
- synthetic data is being used to validate the pipeline,
- modularity is a primary design goal.

So when proposing changes, prefer:
- small local modifications,
- preserved interfaces,
- incremental additions,
- tests for each new block.

---

## Suggested conventions for future contributors

### Naming
Prefer explicit names such as:
- `TabularEncoder`
- `IrregularTSEncoder`
- `SparseMoEFusion`
- `PosteriorHead`
- `TabularDecoder`

### Return dictionaries
For complex forwards, prefer explicit dictionaries over long tuples.

Example:
```python
{
    "tab_emb": ...,
    "ts_emb": ...,
    "fused": ...,
    "mu": ...,
    "logvar": ...,
    "z": ...,
    "tab_recon": ...,
}
```

### Tests
Every new module should ideally get:
- one smoke test,
- one unit test for shape/interface,
- one integration test if it affects training.

---

## How an LLM agent should help on this repository

When answering questions about this repo, the agent should prioritize:
1. understanding module boundaries,
2. preserving the modular architecture,
3. checking tensor and interface compatibility,
4. proposing minimally invasive fixes,
5. keeping new implementations locally pluggable.

The agent should clearly distinguish between:
- **already implemented blocks**
- **partially implemented blocks**
- **natural next extensions**

---

## Summary

`fusemoe_gen` is a modular research MVP for a multimodal VAE-like pipeline using sparse MoE fusion, currently centered on synthetic tabular + irregular time-series data.

Core idea:

```text
data -> modality encoders -> sparse MoE fusion -> posterior -> decoder(s) -> losses -> training
```

Most important repository principle:

> keep each block replaceable without rewriting the whole pipeline
