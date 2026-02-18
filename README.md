# MAcc Exit Survey Rank Ordering Workflow

This repository contains a deterministic analysis workflow that ranks MAcc program/course items from one year of exit survey responses.

## What the workflow does
- Loads the first survey file from `data/` (prefers `.csv`; can fall back to `.xlsx` locally).
- Detects likely course/program rating columns using deterministic heuristics.
- Computes ranking metrics (`mean_rating`, `n_valid`) and tie-breaks deterministically.
- Writes outputs to `outputs/`:
  - `rank_order.csv`
  - `rank_order.png`
  - `reflection.md` (stub template, only created if missing)
- GitHub Actions runs the script on pull requests and on pushes to `main`, and uploads `outputs/*` as an artifact.

## Run locally
```bash
pip install -r requirements.txt
python src/rank_order.py
```

Optional flags:
```bash
python src/rank_order.py --input data/<file>.csv --outdir outputs --topn 10
```

## Ranking rule (brief)
For each detected item column:
1. Calculate `n_valid` (non-null numeric count).
2. Calculate `mean_rating`.
3. Sort by:
   - `mean_rating` descending,
   - then `n_valid` descending,
   - then `item_label` ascending.

This ensures deterministic ordering when scores tie.

## Outputs location
Generated files are stored in `outputs/` and uploaded in Actions as artifact **outputs**.
