---
title: IndoBerea v1
emoji: 📖
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
models:
  - makarios19/IndoBerea-v1
short_description: Reproduction demo of IndoBerea v1 (ICIC 2023)
---

# IndoBerea v1

Reproduction demo for *IndoBerea: Evolving Semantic Search in Theological Context*
(F. V. P. Samosir and S. Mendrofa, ICIC 2023, IEEE; https://doi.org/10.1109/ICIC60109.2023.10382053).

- Code, known issue, and research roadmap: https://github.com/FeliksMakarios/IndoBerea
- Model: https://huggingface.co/makarios19/IndoBerea-v1

The index holds sentence embeddings of the Indonesian Terjemahan Baru (TB) Bible and verse
references only. The verse text is not included because TB is copyrighted by Lembaga Alkitab
Indonesia; each result links to an online reader instead.

**Known issue.** The v1 training objective produced no gradient, so the model behaves as
IndoBERT-base-p2 with mean pooling. See the GitHub repository for details.
