#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


ARTIFACT_DIR = Path(__file__).resolve().parent
FEATURE_EXTRACTOR_PATH = ARTIFACT_DIR / "feature_extractor_frozen.py"
MODEL_PATH = ARTIFACT_DIR / "model.npz"


def _load_feature_module():
    spec = importlib.util.spec_from_file_location("gen17_stage1_feature_extractor", FEATURE_EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load feature extractor from {FEATURE_EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FEATURE_MODULE = _load_feature_module()


def load_model(model_path: Path = MODEL_PATH) -> Dict[str, object]:
    payload = np.load(model_path, allow_pickle=True)
    feature_names = [str(item) for item in payload["feature_names"].tolist()]
    return {
        "weights": payload["weights"].astype(np.float64),
        "bias": float(payload["bias"][0]),
        "mean": payload["mean"].astype(np.float64),
        "std": payload["std"].astype(np.float64),
        "feature_names": feature_names,
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def featurize_chunk(chunk: Sequence[dict]) -> Dict[str, float]:
    all_hand_features: List[Dict[str, object]] = []
    all_decision_records: List[Dict[str, object]] = []
    for hand_index, hand in enumerate(chunk):
        if not isinstance(hand, dict):
            continue
        hand_features, decision_records = _FEATURE_MODULE.compute_hand_features(hand, hand_index)
        all_hand_features.append(hand_features)
        all_decision_records.extend(decision_records)
    if not all_hand_features:
        raise ValueError("Chunk has no valid hands for scoring")
    return _FEATURE_MODULE.aggregate_chunk_features(all_hand_features, all_decision_records)


def score_chunk(chunk: Sequence[dict], model: Dict[str, object] | None = None) -> float:
    model = model or load_model()
    features = featurize_chunk(chunk)
    feature_vector = np.array([float(features[name]) for name in model["feature_names"]], dtype=np.float64)
    standardized = (feature_vector - model["mean"]) / model["std"]
    probability = float(sigmoid(standardized @ model["weights"] + model["bias"]))
    return probability


def _extract_chunk(payload: object) -> Sequence[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("hands"), list):
            return payload["hands"]
        if isinstance(payload.get("chunk"), list):
            return payload["chunk"]
    raise ValueError("Unsupported payload format; expected a list of hands or object with 'hands'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a chunk with gen17-preprod-stage1")
    parser.add_argument("input", type=Path, help="JSON file containing a chunk payload")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    chunk = _extract_chunk(payload)
    result = {
        "probability_bot": score_chunk(chunk),
        "chunk_hand_count": len(chunk),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())