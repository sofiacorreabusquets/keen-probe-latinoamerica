import argparse
import json
import sys
from datetime import datetime

import numpy as np
import torch
from scipy.stats import pearsonr

import hyperparams
from mlp_regressor import MLPRegressor


# Constants
DEFAULT_MODEL_PATH = "models/mlp_regressor_state_dict.pth"
DEFAULT_METADATA_PATH = "data/score_by_entity.json"
INPUT_SIZE = 3072


def load_model(model_path: str, input_size: int = INPUT_SIZE) -> MLPRegressor:
    """
    Load trained MLPRegressor model from state dict.

    Args:
        model_path: Path to .pth file with model weights
        input_size: Input feature dimension (default 3072)

    Returns:
        MLPRegressor model in eval mode on GPU

    Raises:
        RuntimeError: If CUDA is not available
        FileNotFoundError: If model file does not exist
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This inference script requires a GPU. "
            "Please run on a machine with CUDA support."
        )

    # Instantiate model with dummy hyperparams (not used for inference)
    model = MLPRegressor(
        input_size=input_size,
        optimizer="adam",
        learning_rate=0.01,
        max_iter=100
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.cuda()
    model.eval()

    return model


def load_vectors(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load normalized representations from NPZ file.

    Args:
        npz_path: Path to .npz file

    Returns:
        Tuple of (representations_normalized, row_indices)

    Raises:
        FileNotFoundError: If file does not exist
        KeyError: If expected arrays are not in the file
        ValueError: If vector dimensions don't match expected size
    """
    with np.load(npz_path) as data:
        if "representations_normalized" not in data:
            raise KeyError(
                f"NPZ file missing 'representations_normalized' array. "
                f"Available keys: {list(data.keys())}"
            )
        if "row_indices" not in data:
            raise KeyError(
                f"NPZ file missing 'row_indices' array. "
                f"Available keys: {list(data.keys())}"
            )

        representations = data["representations_normalized"]
        row_indices = data["row_indices"]

    # Validate dimensions
    if representations.shape[1] != INPUT_SIZE:
        raise ValueError(
            f"Expected vectors of dimension {INPUT_SIZE}, "
            f"got {representations.shape[1]}"
        )

    return representations, row_indices


def load_entity_metadata(json_path: str) -> dict[int, dict]:
    """
    Load entity metadata indexed by vector_index.

    Args:
        json_path: Path to score_by_entity.json

    Returns:
        Dict mapping vector_index -> entity info dict

    Raises:
        FileNotFoundError: If file does not exist
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Index by vector_index for O(1) lookup
    metadata = {item["vector_index"]: item for item in data}

    return metadata


def run_inference(
    model: MLPRegressor,
    vectors: np.ndarray,
    batch_size: int = hyperparams.BATCH_SIZE,
    verbose: bool = False
) -> np.ndarray:
    """
    Run batched inference on input vectors.

    Args:
        model: Loaded MLPRegressor model
        vectors: Input array of shape (N, 3072)
        batch_size: Batch size for inference
        verbose: Print progress information

    Returns:
        Predictions array of shape (N,)
    """
    predictions = []
    num_samples = vectors.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(0, num_samples, batch_size):
        batch = vectors[i : i + batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.float32).cuda()

        with torch.no_grad():
            batch_preds = model.predict(batch_tensor)

        predictions.append(batch_preds.cpu().numpy())

        if verbose:
            batch_num = i // batch_size + 1
            print(f"Processed batch {batch_num}/{num_batches}", file=sys.stderr)

    return np.concatenate(predictions).squeeze()


def map_predictions_to_entities(
    predictions: np.ndarray,
    row_indices: np.ndarray,
    entity_metadata: dict[int, dict]
) -> list[dict]:
    """
    Map predictions back to entity information.

    Args:
        predictions: Model predictions of shape (N,)
        row_indices: Vector indices from NPZ file
        entity_metadata: Loaded entity metadata

    Returns:
        List of dicts with entity info and prediction
    """
    results = []

    for pred, idx in zip(predictions, row_indices):
        idx = int(idx)
        entity = entity_metadata.get(idx)

        if entity is None:
            # Entity not found in metadata
            result = {
                "vector_index": idx,
                "id_interno": None,
                "entidad": None,
                "pais": None,
                "prediction": float(pred),
                "ground_truth": None
            }
        else:
            result = {
                "vector_index": idx,
                "id_interno": entity.get("id_interno"),
                "entidad": entity.get("entidad"),
                "pais": entity.get("pais"),
                "prediction": float(pred),
                "ground_truth": entity.get("score")
            }

        results.append(result)

    return results


def compute_metrics(results: list[dict]) -> dict:
    """
    Calculate correlation metrics between predictions and ground truth.

    Args:
        results: List of result dicts with 'prediction' and 'ground_truth' keys

    Returns:
        Dict with pearson_correlation, mse, mae metrics
    """
    # Filter out results without ground truth
    valid_results = [r for r in results if r["ground_truth"] is not None]

    if not valid_results:
        return {
            "pearson_correlation": None,
            "p_value": None,
            "mse": None,
            "mae": None,
            "num_samples": 0
        }

    predictions = np.array([r["prediction"] for r in valid_results])
    ground_truth = np.array([r["ground_truth"] for r in valid_results])

    # Pearson correlation
    pearson_corr, p_value = pearsonr(predictions, ground_truth)

    # MSE and MAE
    mse = float(np.mean((predictions - ground_truth) ** 2))
    mae = float(np.mean(np.abs(predictions - ground_truth)))

    return {
        "pearson_correlation": float(pearson_corr),
        "p_value": float(p_value),
        "mse": mse,
        "mae": mae,
        "num_samples": len(valid_results)
    }


def format_output(
    results: list[dict],
    metrics: dict | None,
    metadata: dict,
    output_format: str
) -> str:
    """
    Format results for output.

    Args:
        results: List of prediction results
        metrics: Computed metrics (or None)
        metadata: Run metadata (paths, timestamp, etc.)
        output_format: 'json', 'csv', or 'jsonl'

    Returns:
        Formatted string
    """
    if output_format == "json":
        output = {
            "metadata": metadata,
            "predictions": results
        }
        if metrics:
            output["metrics"] = metrics
        return json.dumps(output, indent=2, ensure_ascii=False)

    elif output_format == "jsonl":
        lines = [json.dumps(r, ensure_ascii=False) for r in results]
        return "\n".join(lines)

    elif output_format == "csv":
        # CSV header
        headers = ["vector_index", "id_interno", "entidad", "pais", "prediction", "ground_truth"]
        lines = [",".join(headers)]

        for r in results:
            row = [
                str(r.get("vector_index", "")),
                str(r.get("id_interno") or ""),
                f'"{r.get("entidad") or ""}"',  # Quote entity names (may contain commas)
                str(r.get("pais") or ""),
                f"{r.get('prediction', 0):.6f}",
                str(r.get("ground_truth") or "")
            ]
            lines.append(",".join(row))

        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown output format: {output_format}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on the trained MLP probe model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --vectors data/filtered_vectors.npz
  python inference.py --vectors data/filtered_vectors.npz --compare
  python inference.py --vectors data/filtered_vectors.npz --format csv --output results.csv
        """
    )

    parser.add_argument(
        "--vectors",
        required=True,
        help="Path to .npz file with input vectors"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model weights (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help=f"Path to entity metadata JSON (default: {DEFAULT_METADATA_PATH})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=hyperparams.BATCH_SIZE,
        help=f"Batch size for inference (default: {hyperparams.BATCH_SIZE})"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "jsonl"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Include ground truth comparison metrics"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        print(f"Loading model from {args.model}...", file=sys.stderr)

    # Load model
    model = load_model(args.model)

    if args.verbose:
        print(f"Loading vectors from {args.vectors}...", file=sys.stderr)

    # Load vectors
    vectors, row_indices = load_vectors(args.vectors)

    if args.verbose:
        print(f"Loaded {len(vectors)} vectors", file=sys.stderr)
        print(f"Loading entity metadata from {args.metadata}...", file=sys.stderr)

    # Load entity metadata
    entity_metadata = load_entity_metadata(args.metadata)

    if args.verbose:
        print(f"Loaded metadata for {len(entity_metadata)} entities", file=sys.stderr)
        print("Running inference...", file=sys.stderr)

    # Run inference
    predictions = run_inference(
        model, vectors,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    if args.verbose:
        print("Mapping predictions to entities...", file=sys.stderr)

    # Map to entities
    results = map_predictions_to_entities(predictions, row_indices, entity_metadata)

    # Compute metrics if requested
    metrics = None
    if args.compare:
        if args.verbose:
            print("Computing metrics...", file=sys.stderr)
        metrics = compute_metrics(results)

        if args.verbose and metrics["num_samples"] > 0:
            print(f"Pearson correlation: {metrics['pearson_correlation']:.4f}", file=sys.stderr)
            print(f"MSE: {metrics['mse']:.6f}", file=sys.stderr)
            print(f"MAE: {metrics['mae']:.6f}", file=sys.stderr)

    # Prepare run metadata
    run_metadata = {
        "model_path": args.model,
        "vectors_path": args.vectors,
        "num_predictions": len(results),
        "timestamp": datetime.now().isoformat()
    }

    # Format output
    output = format_output(results, metrics, run_metadata, args.format)

    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        if args.verbose:
            print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
