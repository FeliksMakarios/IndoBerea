---
title: IndoBerea v1
emoji: 📖
colorFrom: indigo
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - makarios19/IndoBerea-v1
short_description: Precomputed results of IndoBerea v1 (ICIC 2023)
---

# IndoBerea v1

Static page showing the results of *IndoBerea: Evolving Semantic Search in Theological Context*
(F. V. P. Samosir and S. Mendrofa, ICIC 2023, IEEE; https://ieeexplore.ieee.org/document/10382053/).

- Code, known issue, and research roadmap: https://github.com/FeliksMakarios/IndoBerea
- Model: https://huggingface.co/makarios19/IndoBerea-v1

## Limitations

- **Static, no live search.** The page shows only the precomputed top-10 results for the seven
  queries in Table I of the paper. Other queries cannot be tried here.
- **The v1 model is effectively untrained.** The v1 training objective produced no gradient, so the
  model behaves as `indobenchmark/indobert-base-p2` with mean pooling. The results document v1 and
  serve as a baseline; they are not a Bible study tool.
- **Relevance judgments** were made by the paper's authors, without inter-annotator agreement.
- **No verse text.** Terjemahan Baru is copyrighted by Lembaga Alkitab Indonesia; each result links
  to an online reader instead.

IndoBerea v2, with a relevance benchmark and properly trained retrievers, is in development.
