import json
import os

import gradio as gr
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("MODEL_ID", "makarios19/IndoBerea-v1")
INDEX_DIR = os.environ.get("INDEX_DIR", "index")
# Verse text is not shipped with this Space; results link to an online reader instead.
VERSE_URL = os.environ.get("VERSE_URL", "https://www.bible.com/bible/306/{usfm}.{chapter}.{verse}.TB")
CHAPTER_URL = os.environ.get("CHAPTER_URL", "https://www.bible.com/bible/306/{usfm}.{chapter}.TB")
REPO_URL = "https://github.com/FeliksMakarios/IndoBerea"

PAPER_QUERIES = [
    "dosa manusia",
    "kasih pada sesama",
    "pengharapan dalam tekanan",
    "tetap bersukacita",
    "menantikan waktu Tuhan",
    "mengampuni sesama",
    "hidup kekal dan hasilnya",
]

refs = pd.read_csv(os.path.join(INDEX_DIR, "refs.csv"))
embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy")).astype(np.float32)
with open(os.path.join(INDEX_DIR, "meta.json")) as f:
    meta = json.load(f)
model = SentenceTransformer(MODEL_ID)

dup_group = refs["dup_group"].to_numpy()
also_in = refs.groupby("dup_group")["ref"].apply(list).to_dict()


def search(query, top_k):
    query = (query or "").strip()
    if not query:
        return "Masukkan kueri terlebih dahulu."
    q = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ q

    # Same rule as the paper: verses with identical text are shown once.
    seen, rows = set(), []
    for i in np.argsort(-scores):
        g = dup_group[i]
        if g in seen:
            continue
        seen.add(g)
        r = refs.iloc[i]
        verse = VERSE_URL.format(usfm=r.usfm, chapter=r.chapter, verse=r.verse)
        chapter = CHAPTER_URL.format(usfm=r.usfm, chapter=r.chapter)
        others = [x for x in also_in[g] if x != r.ref]
        if len(others) > 5:
            others = others[:5] + [f"{len(others) - 5} lainnya"]
        name = r.ref + (f" (juga {', '.join(others)})" if others else "")
        rows.append(f"| {len(rows) + 1} | {name} | {scores[i]:.4f} | [ayat]({verse}) · [pasal]({chapter}) |")
        if len(rows) == top_k:
            break
    header = "| # | Ayat | Skor cosine | Baca |\n|---|---|---|---|\n"
    return header + "\n".join(rows)


INTRO = f"""
# IndoBerea v1

Demo pencarian semantik ayat Alkitab Terjemahan Baru dari paper
*IndoBerea: Evolving Semantic Search in Theological Context* ([Samosir & Mendrofa, ICIC 2023](https://doi.org/10.1109/ICIC60109.2023.10382053)).

**Catatan penting.** Pelatihan pada v1 ternyata tidak menghasilkan sinyal belajar, sehingga model ini
pada dasarnya adalah IndoBERT (`indobenchmark/indobert-base-p2`) dengan *mean pooling*.
Space ini disediakan untuk reproduksi paper dan sebagai pembanding, bukan sebagai alat studi Alkitab.
Penjelasan lengkap dan rencana IndoBerea v2 ada di [repositori GitHub]({REPO_URL}).

Teks ayat tidak ditampilkan karena hak cipta. Klik tautan untuk membaca ayat dan pasalnya.
"""

with gr.Blocks(title="IndoBerea v1") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        query = gr.Textbox(label="Kueri", placeholder="misalnya: tetap bersukacita", scale=4)
        button = gr.Button("Cari", variant="primary", scale=1)
    top_k = gr.Slider(1, 30, value=10, step=1, label="Jumlah hasil")
    results = gr.Markdown()
    gr.Examples(PAPER_QUERIES, inputs=query, label="Kueri dari Tabel I paper")
    gr.Markdown(f"<small>Model: `{MODEL_ID}` · {meta['verses']} ayat · indeks tanpa teks ayat</small>")
    button.click(search, [query, top_k], results)
    query.submit(search, [query, top_k], results)

if __name__ == "__main__":
    demo.launch()
