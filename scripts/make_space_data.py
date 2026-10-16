"""Write space/data.json, the precomputed v1 results shown by the static Space.

Uses only files in results/: the paper's Table I (verse references, human judgments),
the original notebook scores (four decimals), and the reproduction summary.

    python scripts/make_space_data.py
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Book abbreviations used in tb.csv (alkitab.mobi style) that occur in the paper's results.
ABBR_TO_USFM = {
    "Kej": "GEN", "Kel": "EXO", "Ula": "DEU", "Ayb": "JOB", "Mzm": "PSA", "Ams": "PRO",
    "Kid": "SNG", "Yer": "JER", "Rat": "LAM", "Yeh": "EZK", "Hos": "HOS", "2Ta": "2CH",
    "Mat": "MAT", "Luk": "LUK", "Yoh": "JHN", "Rom": "ROM", "1Ko": "1CO", "Gal": "GAL",
    "Flp": "PHP", "1Te": "1TH", "Ibr": "HEB", "1Yo": "1JN", "Yud": "JUD",
}


def read_csv(name):
    with open(os.path.join(ROOT, "results", name), newline="") as f:
        return list(csv.DictReader(f))


def main():
    table1 = read_csv("table1_paper.csv")
    scores = {(r["query"], r["ref"]): float(r["score"]) for r in read_csv("notebook_scores.csv")}
    with open(os.path.join(ROOT, "results", "reproduction_summary.json")) as f:
        reproduction = json.load(f)

    queries = {}
    for r in table1:
        book, cv = r["ref"].split(" ")
        chapter, verse = cv.split(":")
        if book not in ABBR_TO_USFM:
            raise KeyError(f"no USFM code for book abbreviation {book!r}")
        queries.setdefault(r["query"], []).append({
            "rank": int(r["rank"]),
            "ref": r["ref"],
            "usfm": ABBR_TO_USFM[book],
            "chapter": int(chapter),
            "verse": int(verse),
            "score": scores[(r["query"], r["ref"])],
            "human": int(r["human_judgment"]),
        })

    data = {
        "queries": [{"query": q, "results": res} for q, res in queries.items()],
        "reproduction": reproduction,
    }
    out = os.path.join(ROOT, "space", "data.json")
    with open(out, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}: {len(queries)} queries")


if __name__ == "__main__":
    main()
