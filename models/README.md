# gen17-preprod-stage1

Frozen Phase 1 artifact for miner preprod evaluation.

## Contents

- `model.npz` - NumPy logistic-regression weights and preprocessing stats
- `feature_extractor_frozen.py` - frozen Phase 1 chunk feature extractor
- `score_chunk.py` - local scoring entrypoint for raw chunk JSON
- `offline_metrics.json` - offline benchmark metrics for this frozen stage
- `dataset_manifest.json` - dataset export manifest used for training
- `model_manifest.json` - local manifest metadata

## Usage

Score a raw chunk payload stored as either:

- a JSON list of hands, or
- an object with `hands`

Example:

```bash
python score_chunk.py /path/to/chunk.json
```

Returns `probability_bot` for the hero-centric chunk.

## Status

This artifact is suitable for a preprod miner slot.
It is not yet a promoted production baseline.