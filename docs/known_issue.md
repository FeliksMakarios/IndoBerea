# Known issue: the v1 training run did not train the model

**Summary.** The model described in the ICIC 2023 paper is, in practice,
`indobenchmark/indobert-base-p2` with mean pooling. The fine-tuning step on the Indonesian
Bible did not change the model, so the paper's results are those of an IndoBERT encoder that
was never trained for sentence similarity.

## Cause

The original notebook (`notebooks/IndoBerea_original_2023.ipynb`) prepared the training data
and loss as follows:

```python
examples = [InputExample(texts=[sentence]) for sentence in sentences]
train_loss = losses.BatchHardTripletLoss(model=model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)
```

`BatchHardTripletLoss` needs class labels: for each anchor it takes the hardest positive
(same label) and the hardest negative (different label). `InputExample` without a `label`
argument gets label 0, so every verse in every batch had the same label. With no negatives
in the batch:

- the hardest positive distance is the largest distance in the row, `max_j d(i, j)`;
- the "hardest negative" falls back to `d(i, i) + max_j d(i, j)`, which is the same value,
  because `d(i, i) = 0`;
- the loss becomes `max - max + margin = 5.0` for every batch, and its gradient is zero.

AdamW still applies weight decay with zero gradients, which shrinks the weights by well under
0.1% over the whole run. That is too small to change rankings in a meaningful way.

The notebook also describes the step as learning "to predict the next word"; no language
modelling objective was used.

## Evidence

1. `scripts/check_batch_hard_loss.py` reproduces the setup on a tiny BERT: loss 5.0, gradient
   norm 0.0. With varied labels the same loss gives a non-zero gradient.
2. In the original notebook, the query `kasih Allah` returns the same ten verses in the same
   order for IndoBerea-v2 (1 epoch) and IndoBerea-v3 (3 more epochs).
3. Table I of the paper shows a pattern typical of an encoder not trained for similarity:
   very short verses rank high for unrelated queries. Yoh 6:48 ("Akulah roti hidup") appears
   for 6 of the 7 queries and 1Te 5:19 for 4 of them.
4. `scripts/build_v1.py` rebuilds IndoBERT with mean pooling and compares its top-10 results
   and scores with the outputs saved in the original notebook (`results/notebook_scores.csv`).

## Other limitations of the v1 evaluation

- Seven queries, judged by the authors, without inter-annotator agreement.
- No baseline (for example BM25 or a multilingual sentence encoder).
- The "moderate correlation" between mean similarity and mean human judgment rests on seven points.

## What changes

IndoBerea v2 addresses these points with a relevance benchmark annotated by several judges,
standard retrieval baselines, and retrievers trained with objectives that do provide a
learning signal. See the roadmap in the main README.
