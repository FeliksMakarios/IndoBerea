# Reproduction of IndoBerea v1

Output of `scripts/build_v1.py`, run in Google Colab on 2026-10-16.

- Base model: `indobenchmark/indobert-base-p2` with mean pooling
- Corpus: 31,102 TB verses, 30,705 unique texts
- Same verse at the same rank as the original notebook: 94.3% (66/70)
- Mean absolute score difference: 0.0002
- Max absolute score difference: 0.0007

Per query:

| Query | Same rank | Mean abs diff |
|---|---|---|
| dosa manusia | 10/10 | 0.0002 |
| kasih pada sesama | 10/10 | 0.0002 |
| pengharapan dalam tekanan | 9/10 | 0.0002 |
| tetap bersukacita | 10/10 | 0.0002 |
| menantikan waktu Tuhan | 9/10 | 0.0002 |
| mengampuni sesama | 8/10 | 0.0001 |
| hidup kekal dan hasilnya | 10/10 | 0.0004 |

The scores differ by less than 0.001, so the rebuilt model matches the 2023 model. Rank changes
are expected where the original scores are nearly tied (for example Ula 5:18 and Ula 5:17, both
0.5741 for "mengampuni sesama"), since tiny numeric differences can swap their order. The per-row
comparison is written to `build/reproduction.csv` when the script runs.
