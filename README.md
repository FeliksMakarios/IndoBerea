# IndoBerea

Semantic search over the Indonesian Bible (Terjemahan Baru), from the paper

> F. V. P. Samosir and S. Mendrofa, "IndoBerea: Evolving Semantic Search in Theological Context," in *2023 Eighth International Conference on Informatics and Computing (ICIC)*, IEEE, 2023, pp. 1-6. DOI: [10.1109/ICIC60109.2023.10382053](https://doi.org/10.1109/ICIC60109.2023.10382053)

| | Link |
|---|---|
| Results page (static Hugging Face Space) | https://huggingface.co/spaces/makarios19/IndoBerea-v1 |
| Model | https://huggingface.co/makarios19/IndoBerea-v1 |
| Original notebook (2023, outputs removed) | [`notebooks/IndoBerea_original_2023.ipynb`](notebooks/IndoBerea_original_2023.ipynb) |
| Known issue | [`docs/known_issue.md`](docs/known_issue.md) |

> **Known issue.** The v1 training run produced no gradient, so the published model behaves as
> `indobenchmark/indobert-base-p2` with mean pooling. The cause, a runnable check, and the
> consequences for the paper's results are described in [`docs/known_issue.md`](docs/known_issue.md).
> This repository keeps v1 as a reproducible baseline; IndoBerea v2 fixes the method.

## What v1 does

1. Takes the Terjemahan Baru (TB) and Versi Mudah Dibaca (VMD) texts from the Kaggle dataset
   [williammulianto/indonesia-bible-tb](https://www.kaggle.com/datasets/williammulianto/indonesia-bible-tb)
   (31,102 verses each).
2. Encodes every TB verse with a SentenceTransformers model built on IndoBERT-base-p2.
3. For a query, returns the ten verses with the highest cosine similarity, showing verses with
   identical text (e.g. Kel 20:14 and Ula 5:18) only once.

## Reproducing v1

The original weights were lost, but because training did not change them, the model can be
rebuilt exactly enough to reproduce the paper.

```bash
pip install -r requirements.txt

# 1. Download tb.csv from Kaggle (needs your own Kaggle API token)
kaggle datasets download -d williammulianto/indonesia-bible-tb && unzip indonesia-bible-tb.zip

# 2. Rebuild the model and check it against the original notebook
python scripts/build_v1.py --tb-csv tb.csv --out build
cat build/reproduction_report.md

# 3. Optional: publish the model, its card, and the static Space (needs a Hugging Face write token)
HF_TOKEN=... python scripts/push_to_hub.py --user makarios19 --what model card space

# Check the training issue (CPU, a few seconds, no download)
python scripts/check_batch_hard_loss.py
```

[`notebooks/build_and_publish_v1.ipynb`](notebooks/build_and_publish_v1.ipynb) runs the same steps in Google Colab.

`results/table1_paper.csv` holds Table I of the paper (top-10 verses, scores, and human
judgments for seven queries); `results/notebook_scores.csv` holds the same results with four
decimals, taken from the original notebook outputs.

### Reproduction result

Rebuilt in Google Colab on 2026-10-16 ([`results/reproduction_report.md`](results/reproduction_report.md)):
for the seven queries in the paper, the largest score difference from the 2023 notebook is
**0.0007** (mean 0.0002), and **66 of 70** verses appear at the same rank. The remaining
differences are between verses with nearly tied scores.

## Results page (Hugging Face Space)

The [Space](https://huggingface.co/spaces/makarios19/IndoBerea-v1) is a static page with the
precomputed top-10 results, scores, and human judgments for the seven queries in Table I, plus the
reproduction summary. It has **no live search**: v1 is kept as a record and a baseline, and live
search will come with IndoBerea v2. `scripts/make_space_data.py` builds the page data from
`results/`.

## Repository layout

```
notebooks/   original 2023 notebook (outputs removed) and a Colab notebook to rebuild v1
scripts/     rebuild, reproduction check, publishing, and the training-issue check
space/       static Hugging Face Space (results page)
model_card/  model card for the Hugging Face model repo
results/     Table I of the paper, original notebook scores, reproduction result
docs/        known issue
```

## Data and licensing

- Code in this repository: MIT License.
- Bible text is **not** included. TB is copyrighted by Lembaga Alkitab Indonesia and VMD by
  its publisher. The Space shows only verse references and links each result to an online
  reader.

## Roadmap: IndoBerea v2

IndoBerea v2 turns this exploratory study into a reproducible retrieval study. Planned work:

- **A relevance benchmark** for Indonesian Bible search: natural queries from pastors,
  theology students, and lay readers, with graded relevance judgments by several annotators
  and reported inter-annotator agreement.
- **Passage-level retrieval**, so that results can be whole pericopes or verse windows rather
  than single verses, as suggested in the v1 paper.
- **Retrievers trained with a real learning signal**, using openly licensed supervision
  (parallel translations, cross-references, topical indexes) instead of unlabeled verses.
- **Standard baselines and metrics**: lexical (BM25) and multilingual neural retrievers,
  evaluated with nDCG@10, Recall@k, and MRR, with significance testing.
- **Openly licensed outputs** where the source texts allow it: dataset, models, and a v2 demo.

The v2 results are planned for submission to an ACL venue. This section will link to them when
they are public.

## Citation

```bibtex
@inproceedings{samosir2023indoberea,
  title     = {IndoBerea: Evolving Semantic Search in Theological Context},
  author    = {Samosir, Feliks Victor Parningotan and Mendrofa, Serius},
  booktitle = {2023 Eighth International Conference on Informatics and Computing (ICIC)},
  publisher = {IEEE},
  pages     = {1--6},
  year      = {2023},
  doi       = {10.1109/ICIC60109.2023.10382053}
}
```

## Ringkasan (Bahasa Indonesia)

IndoBerea adalah model pencarian semantik untuk Alkitab Terjemahan Baru. Repositori ini
menyimpan versi pertama (v1) dari paper ICIC 2023 (IEEE Xplore) agar dapat direproduksi. Pelatihan v1 ternyata
tidak mengubah bobot model, sehingga v1 setara dengan IndoBERT dengan *mean pooling*
(lihat [`docs/known_issue.md`](docs/known_issue.md)). IndoBerea v2 sedang dikembangkan dengan
benchmark relevansi yang dianotasi beberapa penilai, pencarian tingkat perikop, dan model yang
dilatih dengan benar. Teks Alkitab tidak disertakan karena hak cipta. Space v1 berupa halaman
statis yang hanya menampilkan hasil tujuh kueri di paper, tanpa pencarian langsung.

## Authors

- Feliks Victor Parningotan Samosir, Universitas Pelita Harapan ([ORCID 0009-0006-0364-0275](https://orcid.org/0009-0006-0364-0275))
- Serius Mendrofa, Sekolah Tinggi Alkitab Tiranus
