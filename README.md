# Poker44-ml17_pre1

Minimal release repository for Poker44 miner runtime scoring.

This repository is a standalone miner variant prepared for gen17 preprod rollout.

## Quick start

```bash
git clone https://github.com/tomkaba/poker44-miner-ml17_pre1.git
cd poker44-miner-ml17_pre1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run Miner

```bash
python neurons/miner.py
```

or legacy wrapper:

```bash
./start_miner.sh HOTKEY_ID[,HOTKEY_ID2,...]
```

## Implementation

- Scorer entrypoint: poker44/miner_heuristics.py
- Frozen artifact scorer: models/score_chunk.py
- Entry point: neurons/miner.py
- Runtime model: models/model.npz
- Frozen feature extractor: models/feature_extractor_frozen.py

Manifest implementation SHA256 is computed from:

- models/model.npz
- models/score_chunk.py
- models/feature_extractor_frozen.py
- neurons/miner.py
- poker44/miner_heuristics.py
- runtime files tracked in repository
