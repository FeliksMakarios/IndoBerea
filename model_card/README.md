---
language:
  - id
license: mit
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: indobenchmark/indobert-base-p2
tags:
  - sentence-transformers
  - semantic-search
  - bible
  - indonesian
---

# IndoBerea v1

Rebuilt model for *IndoBerea: Evolving Semantic Search in Theological Context*
(F. V. P. Samosir and S. Mendrofa, IEEE, 2024): semantic search over the Indonesian
Terjemahan Baru (TB) Bible.

- Demo: https://huggingface.co/spaces/makarios19/IndoBerea-v1
- Code, known issue, and roadmap: https://github.com/FeliksMakarios/IndoBerea

## What this model is

`indobenchmark/indobert-base-p2` followed by mean pooling, loaded with SentenceTransformers.

The original weights from 2023 were lost. They do not need to be recovered, because the
original training run produced no updates: the notebook built `InputExample(texts=[verse])`
without labels and trained with `BatchHardTripletLoss`. With one label for every example,
that loss is constant and its gradient is zero (a runnable check is in the GitHub repository).
The only change to the weights during that run came from AdamW weight decay, which shrinks
them by well under 0.1%.

`scripts/build_v1.py` in the repository rebuilds this model and compares its top-10 results
and cosine scores with the outputs saved in the original notebook.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("makarios19/IndoBerea-v1")
emb = model.encode(["tetap bersukacita", "Bersukacitalah senantiasa."], normalize_embeddings=True)
print(emb[0] @ emb[1])
```

## Intended use and limitations

- Use this model to reproduce the v1 paper and as a baseline. It is not trained for sentence
  similarity and is not recommended as a Bible study tool.
- Short verses (for example Yoh 6:48 or 1Te 5:19) appear for many unrelated queries.
- Results are retrieval aids, not interpretations; always read the surrounding passage.

IndoBerea v2, with a relevance benchmark and properly trained retrievers, is in progress.

## Citation

```bibtex
@inproceedings{samosir2024indoberea,
  title     = {IndoBerea: Evolving Semantic Search in Theological Context},
  author    = {Samosir, Feliks Victor Parningotan and Mendrofa, Serius},
  booktitle = {TODO: conference name},
  publisher = {IEEE},
  year      = {2024},
  doi       = {TODO}
}
```
