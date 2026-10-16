"""Shared helpers for rebuilding IndoBerea v1 (Samosir & Mendrofa, ICIC 2023)."""
import numpy as np
import pandas as pd

BASE_MODEL = "indobenchmark/indobert-base-p2"

# Protestant canon order, matching the order of books in tb.csv.
USFM_BOOKS = (
    "GEN EXO LEV NUM DEU JOS JDG RUT 1SA 2SA 1KI 2KI 1CH 2CH EZR NEH EST JOB PSA PRO "
    "ECC SNG ISA JER LAM EZK DAN HOS JOL AMO OBA JON MIC NAM HAB ZEP HAG ZEC MAL "
    "MAT MRK LUK JHN ACT ROM 1CO 2CO GAL EPH PHP COL 1TH 2TH 1TI 2TI TIT PHM HEB JAS "
    "1PE 2PE 1JN 2JN 3JN JUD REV"
).split()
assert len(USFM_BOOKS) == 66

# The seven queries reported in Table I of the paper.
PAPER_QUERIES = [
    "dosa manusia",
    "kasih pada sesama",
    "pengharapan dalam tekanan",
    "tetap bersukacita",
    "menantikan waktu Tuhan",
    "mengampuni sesama",
    "hidup kekal dan hasilnya",
]


def build_model(base_model=BASE_MODEL):
    """IndoBERT with mean pooling: what v1 effectively was (see docs/known_issue.md)."""
    from sentence_transformers import SentenceTransformer, models

    transformer = models.Transformer(base_model)
    get_dim = getattr(transformer, "get_embedding_dimension", None) or transformer.get_word_embedding_dimension
    dim = get_dim()
    pooling = models.Pooling(dim, pooling_mode="mean")
    return SentenceTransformer(modules=[transformer, pooling])


def load_tb(path):
    """Read tb.csv from the Kaggle dataset williammulianto/indonesia-bible-tb."""
    df = pd.read_csv(path)
    missing = {"kitab", "pasal", "ayat", "firman"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["firman"] = df["firman"].fillna("").astype(str)
    df["ref"] = df["kitab"] + " " + df["pasal"].astype(str) + ":" + df["ayat"].astype(str)

    books = list(dict.fromkeys(df["kitab"]))
    if len(books) != 66:
        raise ValueError(f"expected 66 books, found {len(books)}")
    df["usfm"] = df["kitab"].map(dict(zip(books, USFM_BOOKS)))

    # Verses with identical text (e.g. Kel 20:14 and Ula 5:18) share a group id,
    # so results can be de-duplicated without shipping the text itself.
    df["dup_group"] = df.groupby("firman", sort=False).ngroup()
    return df.reset_index(drop=True)


def unique_top_k(scores, dup_group, k=10, pool=30):
    """Paper's retrieval rule: take the top `pool` hits, keep the first hit per identical text."""
    order = np.argsort(-scores)[:pool]
    seen, hits = set(), []
    for i in order:
        if dup_group[i] in seen:
            continue
        seen.add(dup_group[i])
        hits.append(int(i))
        if len(hits) == k:
            break
    return hits
