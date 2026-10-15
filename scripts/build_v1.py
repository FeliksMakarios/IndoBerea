"""Rebuild IndoBerea v1, check it against the paper, and write the Space index.

Usage (e.g. in Google Colab, after downloading tb.csv from Kaggle):
    python scripts/build_v1.py --tb-csv tb.csv --out build

Outputs:
    build/model/                  SentenceTransformer model to publish as the v1 model repo
    build/space_index/            embeddings (float16) and verse references, without verse text
    build/reproduction_report.md  comparison with the scores in the original notebook
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1_common import BASE_MODEL, build_model, load_tb, unique_top_k  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reproduce(model, df, emb, reference_csv):
    ref = pd.read_csv(reference_csv)
    dup = df["dup_group"].to_numpy()
    index_of = {r: i for i, r in enumerate(df["ref"])}
    rows = []
    for query, expected in ref.groupby("query", sort=False):
        q = model.encode(query, normalize_embeddings=True)
        scores = emb @ q
        hits = unique_top_k(scores, dup)
        got = [df.at[i, "ref"] for i in hits]
        for _, e in expected.iterrows():
            i = index_of.get(e["ref"])
            rows.append({
                "query": query,
                "rank": int(e["rank"]),
                "ref_paper": e["ref"],
                "ref_rebuilt": got[int(e["rank"]) - 1] if int(e["rank"]) <= len(got) else "",
                "score_paper": float(e["score"]),
                "score_rebuilt": float(scores[i]) if i is not None else float("nan"),
            })
    out = pd.DataFrame(rows)
    out["abs_diff"] = (out["score_paper"] - out["score_rebuilt"]).abs()
    out["same_ref"] = out["ref_paper"] == out["ref_rebuilt"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tb-csv", required=True)
    p.add_argument("--out", default="build")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--reference", default=os.path.join(ROOT, "results", "notebook_scores.csv"))
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    df = load_tb(args.tb_csv)
    print(f"{len(df)} verses, {df['dup_group'].nunique()} unique texts")

    model = build_model(args.base_model)
    model_dir = os.path.join(args.out, "model")
    model.save(model_dir)
    shutil.copy(os.path.join(ROOT, "model_card", "README.md"), os.path.join(model_dir, "README.md"))

    emb = model.encode(
        df["firman"].tolist(), batch_size=args.batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)

    index_dir = os.path.join(args.out, "space_index")
    os.makedirs(index_dir, exist_ok=True)
    np.save(os.path.join(index_dir, "embeddings.npy"), emb.astype(np.float16))
    df[["ref", "usfm", "pasal", "ayat", "dup_group"]].rename(
        columns={"pasal": "chapter", "ayat": "verse"}
    ).to_csv(os.path.join(index_dir, "refs.csv"), index=False)
    with open(os.path.join(index_dir, "meta.json"), "w") as f:
        json.dump({"model": args.base_model, "pooling": "mean", "normalized": True,
                   "corpus": "Terjemahan Baru (TB), text not included", "verses": len(df)}, f, indent=2)

    rep = reproduce(model, df, emb, args.reference)
    rep.to_csv(os.path.join(args.out, "reproduction.csv"), index=False)
    same = rep["same_ref"].mean()
    max_diff = rep["abs_diff"].max()
    lines = [
        "# Reproduction of IndoBerea v1",
        "",
        f"- Base model: `{args.base_model}` with mean pooling",
        f"- Same verse at the same rank as the original notebook: {same:.1%} ({rep['same_ref'].sum()}/{len(rep)})",
        f"- Mean absolute score difference: {rep['abs_diff'].mean():.4f}",
        f"- Max absolute score difference: {max_diff:.4f}",
        "",
        "Per query:",
        "",
        "| Query | Same rank | Mean abs diff |",
        "|---|---|---|",
    ]
    for q, g in rep.groupby("query", sort=False):
        lines.append(f"| {q} | {g['same_ref'].sum()}/{len(g)} | {g['abs_diff'].mean():.4f} |")
    with open(os.path.join(args.out, "reproduction_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    if same < 0.9 or max_diff > 0.01:
        print("\nWARNING: rebuilt model differs from the original notebook more than expected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
