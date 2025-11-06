"""
Utility functions for REfrag.

This module provides common utility functions used across different REfrag components,
including logging setup, timing utilities, and helper functions.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("refrag")


@contextmanager
def timer(name: str, logger: logging.Logger | None = None):
    """
    Context manager for timing code blocks.

    Usage:
        with timer("model_loading"):
            model = load_model()

    Args:
        name: Name of the operation being timed
        logger: Optional logger to log timing information

    Yields:
        Dictionary that will contain the elapsed time
    """
    result = {"elapsed": 0.0}
    start_time = time.time()

    if logger:
        logger.debug(f"Starting: {name}")

    try:
        yield result
    finally:
        elapsed = time.time() - start_time
        result["elapsed"] = elapsed

        if logger:
            logger.info(f"Completed: {name} ({elapsed:.3f}s)")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def batch_cosine_similarity(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and multiple vectors.

    Args:
        query_vec: Query vector of shape (dim,)
        vectors: Matrix of vectors of shape (n, dim)

    Returns:
        Array of cosine similarity scores of shape (n,)
    """
    # Normalize query vector
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    # Normalize all vectors
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    # Compute dot product
    similarities = np.dot(vectors_norm, query_norm)

    return similarities


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get torch device based on availability.

    Args:
        device_str: Device string ("auto", "cuda", "mps", "cpu")

    Returns:
        torch.device instance
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_size(size_bytes: int) -> str:
    """
    Format byte size into human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Input text
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class MetricsTracker:
    """
    Track and compute metrics for REfrag performance.

    This class helps track various metrics during REfrag execution,
    including token counts, timing information, and compression ratios.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: dict[str, list[float]] = {}
        self.start_times: dict[str, float] = {}

    def start(self, metric_name: str) -> None:
        """
        Start timing a metric.

        Args:
            metric_name: Name of the metric to time
        """
        self.start_times[metric_name] = time.time()

    def end(self, metric_name: str) -> float:
        """
        End timing a metric and record the elapsed time.

        Args:
            metric_name: Name of the metric

        Returns:
            Elapsed time in seconds
        """
        if metric_name not in self.start_times:
            raise ValueError(f"Metric '{metric_name}' was not started")

        elapsed = time.time() - self.start_times[metric_name]
        self.record(metric_name, elapsed)
        del self.start_times[metric_name]
        return elapsed

    def record(self, metric_name: str, value: float) -> None:
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_average(self, metric_name: str) -> float:
        """
        Get average value of a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Average value
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 0.0
        return np.mean(self.metrics[metric_name])

    def get_total(self, metric_name: str) -> float:
        """
        Get total sum of a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Total sum
        """
        if metric_name not in self.metrics:
            return 0.0
        return sum(self.metrics[metric_name])

    def get_last(self, metric_name: str) -> float:
        """
        Get last recorded value of a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Last recorded value
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 0.0
        return self.metrics[metric_name][-1]

    def get_summary(self) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for all metrics.

        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        for metric_name, values in self.metrics.items():
            if len(values) > 0:
                summary[metric_name] = {
                    "count": len(values),
                    "total": sum(values),
                    "average": np.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "std": np.std(values) if len(values) > 1 else 0.0,
                }
        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()


def pretty_print_metrics(metrics: dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print metrics dictionary.

    Args:
        metrics: Dictionary of metrics
        indent: Indentation level
    """
    indent_str = "  " * indent
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            pretty_print_metrics(value, indent + 1)
        elif isinstance(value, float):
            print(f"{indent_str}{key}: {value:.4f}")
        else:
            print(f"{indent_str}{key}: {value}")
